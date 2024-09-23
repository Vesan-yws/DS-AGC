# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:11:59 2021

@author: mindlab
"""
import torch.nn as nn
import torch
import math
class SelfAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `queries` 的形状：(`batch_size`, 查询的个数, `d`)
    # `keys` 的形状：(`batch_size`, “键－值”对的个数, `d`)
    # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
    # `valid_lens` 的形状: (`batch_size`,) 或者 (`batch_size`, 查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

# =============================================================================
# 未引入Multi-head机制前：X[batch_size,seq_len,feature_dim]
# 引入head后：X[batch_size*head_num,seq_len,feature_dim/head_num]
# 定义 transpose_qkv()，tanspose_output() 函数实现上述转换：
# =============================================================================
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)#CausalConv1d(1, 1, kernel_size=7, stride=1)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)#CausalConv1d(1, 1, kernel_size=7, stride=1)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)#CausalConv1d(1, 1, kernel_size=7, stride=1)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)


    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# =============================================================================
# batch_size, num_queries, num_hiddens, num_heads  = 2, 4, 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans.shape)
# 
# attention = SelfAttention(dropout=0.5)
# batch_size, num_queries, num_hiddens  = 2, 4, 10
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans)
# =============================================================================
