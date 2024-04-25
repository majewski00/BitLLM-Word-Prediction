import math
from typing import Optional, Literal, Tuple

import torch
import sentencepiece as spm

from bit_llm import LLM


class WordSuggestionWrapper:
    def __init__(
            self,
            model: LLM,
            tokenizer: spm.SentencePieceProcessor,
            temperature: Optional[float] = 0.8,
            vary_temperature: Literal[-1, 0, 1] = 0,
            k: Optional[int] = 2,
            max_depth: Optional[int] = 10,
    ):
        """
        LLM model wrapper that allow easy 'next word prediction' process. Is based on modified Beam Search algorithm that search for the complement of the word or the next word, depending on the input string.

        Args:
            model: LLM instance that will be responsible for generations.
            tokenizer: SentencePieceProcessor tokenizer instance.
            temperature: Temperature value. Will change the confidence of the model, making it more confident with smaller temperature.
            vary_temperature: Will allow temperature changes on every Beam Search step. Value 1 will increase the temperature with searching depth and -1 will decrease it. Default is 0 - no temperature variations.
            k: Parameter indicating how many best indices will be taken at each Beam Search step.
            max_depth: How deep the Beam Search will look. It's separated from 'k' to successfully find longer words with smaller k parameter.

        """

        self.model = model.model
        self.top_p = model._sample_top_p
        self.tokenizer = tokenizer

        if temperature <= 0:
            raise ValueError(
                f"Temperature value in WordSuggestionWrapper has to be larger than 0! Received t={temperature}. "
            )
        self.temperature = temperature
        self.vary_temperature = vary_temperature
        self.k = k
        self.max_depth = max_depth

        self._search_results = []

    def _check_for_space(self, tokens: torch.Tensor):
        """Check if main condition is satisfy -- find 'space' in generated tokens. It means that searching for next word ended. This is essential step when working with tokenizer that samples words into smaller parts. """
        decoded = self.tokenizer.IdToPiece(tokens.tolist())
        for piece in decoded:
            if "▁" in piece:
                return True
        return False

    def _beam_search(
            self,
            input_tokens: torch.Tensor,
            temperature: float = 0.8,
            _rank: int = 1,
            _prob: torch.Tensor = torch.tensor([]),
    ):
        """ Regressive Beam Search algorithm. The algorithm does not return values back, it saves correct generations into _search_results list. """

        ## Check if input tokens have proper size.
        if input_tokens.ndim != 1:
            if input_tokens.shape[0] != 1:
                raise ValueError(
                    "Can generate only one prompt at the time!"
                )
        else:
            input_tokens.unsqueeze_(0)

        logits = self.model(input_tokens)[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)

        if self.max_depth == _rank:
            ## If max rank is reached, select the best token and check if it satisfied the condition.
            out = torch.topk(probs, k=1)
            if self._check_for_space(out.indices.squeeze(0)):
                self._search_results.append([input_tokens.squeeze(0), _prob])
            return

        top_k = torch.topk(probs, k=self.k)
        next_tokens, probs = top_k.indices, top_k.values

        if self.vary_temperature == 1:
            temperature += self.temperature / self.max_depth
        elif self.vary_temperature == -1:
            temperature -= self.temperature / self.max_depth

        for token, prob in zip(next_tokens.T, probs.T):
            if not self._check_for_space(token):
                token = torch.cat((input_tokens, token.unsqueeze(0)), dim=-1)
                self._beam_search(
                    token, temperature, _rank + 1, torch.cat((_prob, prob))
                )
            else:
                self._search_results.append([input_tokens.squeeze(0), _prob])
                break
        return

    @torch.inference_mode()
    def next_word(self, input_sentence: str, n_best: Optional[int] = 10) -> Tuple[str, list, list]:
        """
        Main function of the WordSuggestionWrapper instance. Allows to get probabilities of at most n_best generated tokens.

        Args:
            input_sentence: String that will become an input to LlaMA instance. Generated output will be based on this.
            n_best: Number of the best possible tokens.

        Return:
            String that visually shows the generations and probabilities; Sorted probabilities of tokens; Word Suggestions - generated tokens, correspondent to scores


        """
        encoded_tokens = self.tokenizer.Encode(input_sentence, add_bos=True, enable_sampling=True, alpha=0.2)
        if input_sentence[-1] == " ":
            ## If the input sentence ends with space, add scape_id to tokens. Tokenizer does not do it on its own. This way model will be sure to generate new word, instead of appending tokens to last word, or even generating space at first step.
            space_id = self.tokenizer.PieceToId(['▁'])[0]
            encoded_tokens.append(space_id)

        tokens = torch.tensor(encoded_tokens)
        self._beam_search(tokens, self.temperature)

        cum_score = 0
        scores, generations = [], []
        for gen, score in self._search_results:
            new_tokens = gen[tokens.shape[1]:]
            generations.append(self.tokenizer.Decode(new_tokens.tolist()))
            cum_score += score.sum().item()
            scores.append(score.sum().item())

        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        scores = [scores[i] for i in sorted_indices]
        generations = [generations[i] for i in sorted_indices]

        out_string = ''
        out_scores = []
        out_gen = []
        for i in range(min(len(scores), n_best)):
            out_string += f"{i + 1}.  ...{' '.join((input_sentence+generations[i]).split()[-3:])}    Score: {round(scores[i] / cum_score, 2)}\n"
            out_gen.append(generations[i])
            out_scores.append(round(scores[i] / cum_score, 2))

        if i + 1 == n_best:
            out_string += f"{i + 2}.  ...    Score: {round(1 - sum(out_scores) / cum_score, 2)}\n"

        return out_string, scores, generations
