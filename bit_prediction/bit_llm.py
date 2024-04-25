import math
import time
import json
import warnings
from typing import Optional
from os.path import exists, join
from os import makedirs
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from sentencepiece import SentencePieceProcessor


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor):
    _, _seq_len, _, _ = x.shape
    if _seq_len != freqs_complex.shape[0]:
        freqs_complex = freqs_complex[:_seq_len, :]

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    x_rot = x_complex * freqs_complex
    x_rot = torch.view_as_real(x_rot)
    x_rot = x_rot.reshape(*x.shape)

    return x_rot.type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError(
            f"Dimensions for RoPE must be even! But got head_dim: {head_dim}"
        )

    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


@dataclass
class ModelArgs:
    depth: int = 12
    n_heads: int = 8
    kv_heads: int = 4
    dim: int = 4096
    dim_multi: int = 4
    vocab_size: int = -1

    dropout: float = 0.1
    bias: bool = False
    post_act_ln: bool = False
    norm_eps: float = 1e-8
    pad_id: int = 0

    max_seq_len: int = 1024
    device: str = 'cpu'

    binary: bool = False
    ternary: bool = True
    half: bool = False
    ste: bool = True

    def save(self):
        return asdict(self)


class RSMNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight.type_as(x) * self._norm(x.float()).type_as(x)


class LinearBit(nn.Linear):
    def __init__(self, model_args: ModelArgs, *args, **kwargs):
        super(LinearBit, self).__init__(*args, **kwargs)

        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        self.eps = model_args.norm_eps
        self.rsm = RSMNorm(self.in_features, eps=self.eps)
        self.ternary = model_args.ternary
        self.ste = model_args.ste

        if not model_args.ternary and not model_args.binary:
            self.use_linear = True
        else:
            self.use_linear = False

    def _activation_quant(self, x: torch.Tensor):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.type(torch.float32).clamp_(min=1e-4)
        y = (x * scale).round().clamp_(-128, 127) / scale
        return y.type_as(x)

    def _binary_quant(self, x: torch.Tensor):
        scale = x.abs().mean()
        e = x.mean()
        u = (x - e).sign() * scale
        return u

    def _ternary_quant(self, x: torch.Tensor):
        scale = x.abs().mean()

        ## Dequantization applied before matrix multiplication. This could potentially slowdown the process.
        return torch.round(F.tanh(x / (scale + self.eps))) * torch.atanh(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        x_norm = self.rsm(x)

        if not self.use_linear:
            x_quant = x_norm + (self._activation_quant(x_norm) - x_norm).detach()

            if self.ternary:
                if self.ste:
                    w_quant = w + (self._ternary_quant(w) - w).detach()
                else:
                    w_quant = self._ternary_quant(w)

            else:
                if self.ste:
                    w_quant = w + (self._binary_quant(w) - w).detach()
                else:
                    w_quant = self._binary_quant(w)

            y = F.linear(x_quant, w_quant)
        else:
            y = F.linear(x_norm, w)

        return y


class BitAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.q_heads = args.n_heads
        self.kv_heads = args.kv_heads
        self.dim = args.dim

        self.dropout = args.dropout

        if self.q_heads % self.kv_heads != 0:
            raise ValueError(
                f"Number of KV heads has to be multiple of model heads! Received n_heads={self.q_heads}, kv_heads={self.kv_heads}. "
            )

        self.head_dim = args.dim // self.q_heads

        dtype = torch.float32
        if args.half:
            dtype = torch.half

        self.w_q = LinearBit(
            args, in_features=self.dim, out_features=self.dim, bias=args.bias, device=args.device, dtype=dtype
        )

        self.kv_dim = self.dim // self.q_heads * self.kv_heads
        self.w_k = LinearBit(
            args, in_features=self.dim, out_features=self.kv_dim, bias=args.bias, device=args.device, dtype=dtype
        )
        self.w_v = LinearBit(
            args, in_features=self.dim, out_features=self.kv_dim, bias=args.bias, device=args.device, dtype=dtype
        )

        self.w_o = LinearBit(
            args, in_features=self.kv_dim, out_features=self.dim, bias=args.bias, device=args.device, dtype=dtype
        )

    def _gqa_dot(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            scale: Optional[float] = None,
            mask: Optional[torch.Tensor] = None,
            causal: Optional[bool] = None,
    ):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if scale is None:
            scale = query.size(-1) ** 0.5
        query = query / scale

        _bsz, q_head, _seq, _dim = query.shape
        _, k_head, _, _ = key.shape
        _, v_head, _, _ = value.shape

        n_head_groups = q_head // k_head
        if n_head_groups > 1:
            ## Grouped Query Attention
            query = query.view(_bsz, n_head_groups, k_head, _seq, _dim)
            scores = einsum(query, key, "b g h n d, b h s d -> b h n s")
        else:
            scores = torch.matmul(query, key.transpose(2, 3))

        if causal:
            mask = torch.ones(
                (_bsz, _seq, _seq),
                device=query.device,
                dtype=torch.bool,
            ).tril_()

        if mask is not None:
            if mask.ndim == 2:
                mask.unsqueeze_(1).unsqueeze_(1)
            elif mask.ndim == 3:
                mask.unsqueeze_(1)

            scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)

        attention = F.softmax(scores / scale, dim=-1)
        if self.dropout > 0.0:
            attention = F.dropout(attention, p=self.dropout)

        out = torch.matmul(attention, value).transpose(1, 2)

        return out

    def forward(
            self,
            x: torch.Tensor,
            freq_complex: torch.Tensor,
            causal: bool = False,
    ):
        _bsz, _seq_len, _ = x.shape

        # (B, seq_len, q_heads * dim)
        q: torch.Tensor = self.w_q(x)
        # (B, seq_len, dim * kv_dim)
        ## kv_dim = dim // q_heads * kv_heads
        k: torch.Tensor = self.w_k(x)
        v: torch.Tensor = self.w_v(x)

        q = q.view(_bsz, _seq_len, self.q_heads, self.head_dim)
        k = k.view(_bsz, _seq_len, self.kv_heads, self.head_dim)
        v = v.view(_bsz, _seq_len, self.kv_heads, self.head_dim)

        k = apply_rope(k, freq_complex)
        q = apply_rope(q, freq_complex)

        y = self._gqa_dot(
            query=q,
            key=k,
            value=v,
            causal=causal
        ).type_as(x)

        return self.w_o(y.reshape(_bsz, _seq_len, self.kv_dim))


class BitFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.dim_multi * args.dim

        dim = args.dim

        self.dropout = args.dropout
        self.ln = None
        if args.post_act_ln:
            self.ln = RSMNorm(hidden_dim, args.norm_eps)

        dtype = torch.float32
        if args.half:
            dtype = torch.half

        self.ff1 = LinearBit(
            args, in_features=dim, out_features=hidden_dim, bias=args.bias, device=args.device, dtype=dtype
        )
        self.ff2 = LinearBit(
            args, in_features=dim, out_features=hidden_dim, bias=args.bias, device=args.device, dtype=dtype
        )
        self.ff_out = LinearBit(
            args, in_features=hidden_dim, out_features=dim, bias=args.bias, device=args.device, dtype=dtype
        )


    def forward(self, x: torch.Tensor):
        swish = F.silu(self.ff1(x))
        x_V = self.ff2(x)
        x = swish * x_V

        if self.ln is not None:
            x = self.ln(x)

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        return self.ff_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim

        self.attn = BitAttention(args)
        self.attn_norm = RSMNorm(self.dim, args.norm_eps)

        self.ffn = BitFeedForward(args)
        self.ffn_norm = RSMNorm(self.dim, args.norm_eps)


    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        h = x + self.attn(self.attn_norm(x), freq_complex=freqs_complex, causal=True)
        out = h + self.ffn(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.post_act_ln = args.post_act_ln
        self.vocab_size = args.vocab_size
        if self.vocab_size == -1:
            raise ValueError(
                f"Please set the vocab_size parameter in ModelArgs! "
            )
        if args.ternary and args.binary:
            raise ValueError(
                "Received ternary and binary as True in ModelArgs! Only one or none can be true! "
            )

        dtype = torch.float32
        if args.half:
            dtype = torch.half

        self.emb = nn.Embedding(self.vocab_size, args.dim, padding_idx=args.pad_id, dtype=dtype)

        self.layers = nn.ModuleList()
        for _ in range(args.depth):
            self.layers.append(TransformerBlock(args))

        self.output_norm = RSMNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=args.bias, device=args.device, dtype=dtype)

        self.freqs_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            device=args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = None):
        """ Forward function aware for KV_caching. """
        _bsz, _seq_len = tokens.shape
        h = self.emb(tokens)

        if _seq_len == 1 and start_pos is None:
            raise ValueError(
                "For inference mode use start_pos"
            )

        if _seq_len > 1:
            for layer in self.layers:
                h = layer(h, self.freqs_complex)

        else:
            freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
            for layer in self.layers:
                h = layer(h, freqs_complex)

        h = self.output_norm(h)
        h = self.output(h)

        return h


class LLM:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        """

        Args:
            model: Transformer model.
            tokenizer: SentencePiece tokenizer.
            model_args: ModelArgs instance.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    def __call__(self, tokens: torch.Tensor, **kwargs):
        tokens_inp, labels = tokens[:, :-1], tokens[:, 1:]
        logits = self.model(tokens_inp, **kwargs)
        return F.cross_entropy(logits.transpose(1, 2).type(torch.float32), labels)

    def __str__(self):
        parms = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return str(self.model) + f'\nNumber of parameters: {parms}\n'

    def save(self, model_path: str, name: str = "bit_LLM"):

        if not exists(model_path):
            raise ValueError(
                f"Couldn't find {model_path} path! "
            )
        _i = 0
        while 1:
            if exists(join(model_path, name + (f'({_i})' if _i >= 1 else ''))):
                _i += 1
            else:
                if _i >= 1:
                    name += f"({_i})"
                break

        makedirs(join(model_path, name), exist_ok=True)
        torch.save(self.model.state_dict(), join(model_path, name, "model_state.pth"))

        with open(join(model_path, name, "params.json"), "w") as f:
            json.dump(self.model_args.save(), f)

        return

    @staticmethod
    def build(model_path: str, tokenizer: SentencePieceProcessor, model_args: ModelArgs = None):

        if not exists(model_path):
            raise ValueError(
                f"Couldn't find {model_path} path! "
            )

        checkpoint = torch.load(join(model_path, "model_state.pth"), map_location='cpu')

        if model_args is None:
            with open(join(model_path, "params.json"), "r") as f:
                params = json.loads(f.read())

            model_args: ModelArgs = ModelArgs(
                **params
            )

        device = model_args.device
        if device == "cuda" and torch.cuda.device_count() == 0:
            warnings.warn("Cuda didn't detected! Switching model to cpu.", UserWarning)
            model_args.device = 'cpu'
            device = 'cpu'

        model_args.vocab_size = tokenizer.vocab_size()
        model_args.pad_id = tokenizer.pad_id()

        model: Transformer = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=True)

        if device == "cuda":
            model.cuda()

        return LLM(model, tokenizer, model_args)

    @staticmethod
    def create_new(tokenizer: SentencePieceProcessor = SentencePieceProcessor(), model_args: ModelArgs = ModelArgs()):

        model_args.vocab_size = tokenizer.vocab_size()

        if model_args.device == "cuda" and torch.cuda.device_count() == 0:
            warnings.warn("Cuda didn't detected! Switching model to cpu.", UserWarning)
            model_args.device = 'cpu'

        model = LLM(Transformer(model_args), tokenizer, model_args)


        if torch.cuda.device_count() > 0 and model_args.device == 'cuda':
            model.model.cuda()

        return model

    def _sample_top_p(
            self,
            probs: torch.Tensor,
            p: float,
            samples: int = 1,
            return_probs: bool = False,
    ):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=samples)
        next_token = torch.gather(probs_idx, -1, next_token)

        if return_probs:
            token_prob = torch.gather(probs, -1, next_token)
            return next_token, token_prob

        return next_token

    @torch.inference_mode()
    def generate(
            self,
            input_tokens: torch.Tensor,
            max_seq: int,
            eos_token: Optional[int] = None,
            temperature: Optional[float] = 1.,
            top_p: Optional[float] = .9,
    ) -> torch.Tensor:
        """
        Generation functionality. Based on 'top P' algorithm. Able to generate one sequence at the time!

        Args:
            input_tokens: Tokenized tokens in tensor form. Could be either in [seq_len] or [1, seq_len] shape.
            max_seq: Total max size of generation. This include input size as well.
            eos_token: Index of the tokenizer EOS token. Is not set, generation will end when max_seq length is satisfy.
            temperature: Temperature parameter for generations. Results in change of the model confidence, making it more confident with smaller temperature.
            top_p: Main parameter of the 'top P' algorithm.

        Returns:
            torch.Tensor with generated tokens.
        """
        if int(input_tokens.shape[-1]) >= max_seq:
            raise ValueError(
                "Too long prompt!"
            )
        if not 0 <= temperature <= 1:
            raise ValueError(
                f"{temperature} is incorrect value for temperature! Temperature has to be positive number or 0 (disabled)"
            )

        if input_tokens.ndim != 1:
            if input_tokens.shape[0] != 1:
                raise ValueError(
                    "Can generate only one prompt at the time!"
                )
        else:
            input_tokens.unsqueeze_(0)

        _seq_len = input_tokens.shape[-1]
        # input_tokens.shape = [1, seq_len]

        steps = max_seq - int(input_tokens.shape[-1])

        out = input_tokens
        for _ in range(steps):
            with torch.no_grad():
                logits = self.model(out)[:, -1, :]
                ## input -> output: [a,b,c,d] -> [b,c,d,e]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)  # [1, 1]

            out = torch.cat((out, next_token), dim=-1)

            if eos_token is not None:
                if out[0, -1] == eos_token:
                    break

        out = out[0, _seq_len:]
        return out
