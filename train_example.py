import random
import math
import os
import sys
import time
import datetime

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

from sentencepiece import SentencePieceProcessor
import matplotlib.pyplot as plt
from datasets import load_dataset

from bit_prediction import LLM, ModelArgs


BATCH_SIZE = 32
EPOCH = 1
GRADIENT_ACCUMULATE_EVERY = 4
LR = 2e-5
MAX_LR = 2e-4
VALIDATE_EVERY = 50
GENERATE_EVERY = 500
CHECKPOINT = 4000
LOG_FILE = "log.txt"



def compute_eta(start_time, progress: int, total: int) -> str:
    """
    Display Estimated Time Arrival in clear way.

    Args:
        start_time: instance of time.time(), initialize at the begging of the training.
        progress: Current step in training.
        total: Total steps in training.
    Returns:
        Eta value string, in format "hh:mm:ss"
    """
    if progress == 0:
        return "00:00:00"
    current_time = time.time()
    elapsed_time = current_time - start_time
    estimated_total_time = (elapsed_time / progress) * total
    estimated_remaining_time = estimated_total_time - elapsed_time
    finishing_time = current_time + estimated_remaining_time
    hours, rem = divmod(finishing_time - current_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

def encode(examples):
    """
    Dataset encoding, suitable for BookCorpus, among others. Because of the sub-word tokenization, every text can be
    tokenized multiple times. Design for sequence length of 1024.
    """
    encoded_data = []
    for text in examples['text']:
        for _ in range(2):
            encoded = tokenizer.Encode(text, emit_unk_piece=True, enable_sampling=True, alpha=0.15)
            if len(encoded) > 1024:
                r = random.randint(0, len(encoded)-1024)
                encoded_data.append(encoded[r:r+1024])
            else:
                encoded_data.append(encoded)

    return {'input_ids': encoded_data}


def concatenate_texts(examples):
    """BookCorpus is build in many small data-point. To avoid to much PAD tokens, this function will concatenate the dataset"""
    concat_text = [' '.join(examples["text"][i:i+16]) for i in range(0, len(examples["text"]), 16)]
    return {'text': concat_text}


def split_dataset(dataset, val_split=0.1):
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


class CustomDataset(Dataset):
    def __init__(self, tokenized_data, cuda: bool = False):
        self.tokenized_data = tokenized_data
        self.cuda = cuda

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)

        if self.cuda:
            return input_ids.cuda()
        return input_ids


def main():
    tokenizer = SentencePieceProcessor()
    # tokenizer.load("models/tokenizer.model")


    ## Initialize Large Language Model
    args = ModelArgs(
        depth=4,
        dim=1024,
        n_heads=16,
        kv_heads=4,
        ternary=True,
        binary=False,
        pad_id=tokenizer.pad_id(),
        half=False,
        device='cuda',
    )

    llm = LLM.create_new(tokenizer=tokenizer, model_args=args)
    # print(llm)


    ## Load and preprocess BookCorpus dataset
    dataset = load_dataset("bookcorpus", cache_dir="", num_proc=4)
    dataset = dataset.map(concatenate_texts, batched=True, num_proc=os.cpu_count())
    dataset = dataset.map(
        encode, batched=True, remove_columns=list(dataset.column_names['train']), num_proc=os.cpu_count()
    )
    train_dataset, val_dataset = split_dataset(dataset['train'], 0.05)

    train_dataset = CustomDataset(train_dataset, cuda=True if args.device == 'cuda' else False)
    val_dataset = CustomDataset(val_dataset, cuda=True if args.device == 'cuda' else False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


    ## Optimizer and Scheduler initialization
    step_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    optimizer = torch.optim.AdamW(
        llm.model.parameters(),
        lr=LR,
        eps=1e-8 if not args.half else 1e-4,
        weight_decay=0.25
    )
    clr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=LR,
        max_lr=MAX_LR,
        cycle_momentum=False,
        mode='exp_range',
        gamma=0.99999,
        step_size_up=1000,
        step_size_down=1000
    )


    ## Create log file for learning outputs and dictionary to store loss values
    with open(LOG_FILE, "w") as f:
        f.write('')
    data = {
        "loss": [[], []],
        "val_loss": [[], []]
    }


    ## Training Loop
    start_time = time.time()
    for epoch in range(EPOCH):
        train_loss = 0.
        val_loss = "N/A"
        data_batch = iter(train_loader)
        val_data_batch = iter(val_loader)

        for step in range(step_per_epoch):
            llm.model.train()

            loss = llm(next(data_batch))
            loss.backward()
            clr_scheduler.step()

            if step % GRADIENT_ACCUMULATE_EVERY == 0:
                train_loss = loss.mean().detach().item()
                data['loss'][0].append(step + (step_per_epoch * epoch))
                data['loss'][1].append(train_loss)
                torch.nn.utils.clip_grad_norm_(llm.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            text = f"[{epoch + 1}/{EPOCH}]  {step + 1}/{step_per_epoch} == {round((step + 1) / step_per_epoch * 100, 3)}%    " \
                   f"ETA: {compute_eta(start_time, step + 1 + (step_per_epoch * epoch), step_per_epoch * EPOCH)}      " \
                   f"Training loss: {round(train_loss, 4)}  Validation loss: {val_loss}  "

            if step % VALIDATE_EVERY == 0:
                with torch.no_grad():
                    llm.model.eval()

                    total_val_loss = 0.0
                    n_val = 8
                    for _ in range(n_val):
                        try:
                            loss = llm(next(val_data_batch))
                        except StopIteration:
                            val_data_batch = iter(val_loader)
                            loss = llm(next(val_data_batch))

                        total_val_loss += loss.mean().detach().item()

                    val_loss = round(total_val_loss / n_val, 4)
                    data['val_loss'][0].append(step + (step_per_epoch * epoch))
                    data['val_loss'][1].append(val_loss)



            if step % GENERATE_EVERY == 0:
                try:
                    tokens = next(val_data_batch)[0, :40]
                except StopIteration:
                    val_data_batch = iter(val_loader)
                    tokens = next(val_data_batch)[0, :40]

                gen = llm.generate(tokens, tokens.shape[-1] + 40)
                text += f"\n{step + 1} Step GENERATION\nPrompt: {tokenizer.decode(tokens.tolist())[0]} ** {tokenizer.decode(gen.tolist())} **\n"


            if step % 10 == 0:
                with open(log_name, "r") as f:
                    existing_content = f.read()
                with open(log_name, "w") as f:
                    f.write(f"""'{datetime.datetime.now().strftime("%H:%M:%S")}'    {text}\n""" + existing_content)


            if step % CHECKPOINT == 0:
                llm.save("checkpoint/", name=f"bit_{step + 1}")




if __name__ == "__main__":
    main()




