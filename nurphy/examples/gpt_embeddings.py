import torch
import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    def forward(self,
                x: torch.Tensor,
                transposed: bool = False) -> torch.Tensor:
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))
        else:
            return super().forward(x)


class PositionalEmbedding(nn.Embedding):
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)

        print("***********************")
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        return super().forward(position)


seq_len = 5
dims = 10
dropout = 0.1
words = 100
positional_embedding = PositionalEmbedding(seq_len, dims)
token_embedding = TokenEmbedding(words, dims)
dropout_embedding = nn.Dropout(dropout)

ids = torch.randn((3,4))
s_len = ids.size(-1)
x = torch.arange(s_len, dtype=torch.long, device=ids.device)
print(x)

offset = 0
# Use token embedding and positional embedding layers.
x = token_embedding(x) + positional_embedding(x, offset)
x = dropout_embedding(x)

print(x)