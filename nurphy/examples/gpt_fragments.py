import h5py
import random

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, dataset_path: str, seq_len: int, window_size: int, rng: random.Random):
        self.dataset = h5py.File(dataset_path, 'r')['token_ids']
        self.seq_len = seq_len
        self.window_size = window_size
        self.rng = rng
        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start_point = self.rng.randint(0, self.size)
        end_point = min(start_point + (self.window_size + 1), self.size)
        window = self.dataset[start_point: end_point]  # Windowing

        input_ids = window[:-1]
        label_ids = window[1:]
        return input_ids, label_ids


def create_future_mask(x: torch.Tensor, offset: int = 0) -> torch.Tensor:
    seq_len = x.size(1)  # seq_length
    print(seq_len)

    # Create shifted upper triangular matrix.
    future = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
    future = future.triu(1)

    future_mask = future.view((1,) * (x.ndim - 1) + future.size())
    return future_mask.expand(x.shape + future_mask.shape[-1:])  # (b, s, s)

input_ids = torch.randn((2,10))
future_mask = create_future_mask(input_ids)
print(future_mask)

import torch.nn as nn
vocab_size = 7
hidden_size = 50
token_embeddings_1 = nn.Embedding(vocab_size, hidden_size)
print(token_embeddings_1.weight)

print(token_embeddings_1(torch.tensor(0)))
result = torch.matmul(torch.tensor([1,0,0,0,0,0,0]).float(),token_embeddings_1.weight)
print(result)

input_ids1 = torch.randn((2,vocab_size))
token_embeddings = nn.Embedding(vocab_size, hidden_size)
#token_embeddings = token_embeddings(input_ids1).to(torch.long)

def position_embed(input_ids, position_embeddings):
    seq_length = input_ids.size(-1)
    pos_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    print(pos_ids)
    pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
    print(pos_ids)
    pos_embeddings = position_embeddings(pos_ids)

    return pos_embeddings


pos_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
input_ids = torch.randn((2,vocab_size))
pos_embedding_vec = position_embed(input_ids,pos_embeddings)

print(pos_embedding_vec)


