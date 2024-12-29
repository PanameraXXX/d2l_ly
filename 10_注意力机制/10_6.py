import math
import torch
from torch import nn
from d2l import torch as d2l

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()

# MultiHeadAttention(
#   (attention): DotProductAttention(
#     (dropout): Dropout(p=0.5, inplace=False)
#   )
#   (W_q): Linear(in_features=100, out_features=100, bias=False)
#   (W_k): Linear(in_features=100, out_features=100, bias=False)
#   (W_v): Linear(in_features=100, out_features=100, bias=False)
#   (W_o): Linear(in_features=100, out_features=100, bias=False)
# )

# # 跟我想的一样
# w_o 是一个线性层，用于将多头注意力的输出拼接后转换为期望的维度。它的参数是可学习的矩阵 W_0。
# 1 你可以参考这篇文章2，了解更多关于多头注意力的原理和应用。

# 自注意力机制，无视长度，看的可以很长，从它的机制来看
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)


#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)