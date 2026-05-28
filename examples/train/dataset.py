from torch.utils.data import Dataset
import torch
import numpy as np
from datasets import load_dataset


def compute_max_len(file_path, tokenizer):
    ds = load_dataset("json", data_files=file_path, split="train")
    max_len = 0
    for item in ds:
        tokens = tokenizer.encode(item["Text"]).ids
        max_len = max(max_len, len(tokens))
    return max_len


class PretrainDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.ds = load_dataset("json", data_files=file_path, split="train")
        if max_length is None:
            self.max_length = compute_max_len(file_path, tokenizer)
        else:
            self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["Text"]
        tokens = self.tokenizer.encode(text).ids
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        pad_id = 50257  # GPT2 vocab_size，即新增的 PAD token id
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [pad_id] * (self.max_length - len(input_ids))
        if len(target_ids) < self.max_length:
            target_ids = target_ids + [pad_id] * (self.max_length - len(target_ids))

        input_ids = torch.tensor(input_ids[:self.max_length], dtype=torch.long)
        target_ids = torch.tensor(target_ids[:self.max_length], dtype=torch.long)
        return input_ids, target_ids