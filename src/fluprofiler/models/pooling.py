"""
Pooling layers and attention mechanisms for fluProfiler models.

Contains various pooling implementations including attention-based pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class attention_mask(nn.Module):
    """GlobalMaskValueAttentionPooling1D"""
    def __init__(self, embed_size, units=None, use_additive_bias=False, use_attention_bias=False):
        super(attention_mask, self).__init__()
        self.embed_size = embed_size
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.units = units if units else embed_size
        self.U = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.V = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.W = nn.Parameter(torch.Tensor(self.units, self.embed_size))

        nn.init.trunc_normal_(self.U, std=0.01)
        nn.init.trunc_normal_(self.V, std=0.01)
        nn.init.trunc_normal_(self.W, std=0.01)

        if self.use_additive_bias:
            self.b1 = nn.Parameter(torch.Tensor(self.units))
            nn.init.trunc_normal_(self.b1, std=0.01)
        if self.use_attention_bias:
            self.b2 = nn.Parameter(torch.Tensor(self.embed_size))
            nn.init.trunc_normal_(self.b2, std=0.01)

    def forward(self, x, mask=None, save_attention_path=None):
        # (B, Len, Embed) x (Embed, Units) = (B, Len, Units)
        q = torch.matmul(x, self.U)
        k = torch.matmul(x, self.V)

        if self.use_additive_bias:
            h = torch.tanh(q + k + self.b1)
        else:
            h = torch.tanh(q + k)

        # (B, Len, Units) x (Units, Embed) = (B, Len, Embed)
        if self.use_attention_bias:
            e = torch.matmul(h, self.W) + self.b2
        else:
            e = torch.matmul(h, self.W)

        if mask is not None:
            attention_probs = nn.Softmax(dim=1)(e + torch.unsqueeze((1.0 - mask) * -10000, dim=-1))
        else:
            attention_probs = nn.Softmax(dim=1)(e)

        if save_attention_path is not None:
            filenames = os.listdir(save_attention_path)
            if len(filenames) == 0:
                max_num = 0
            else:
                max_num = max([int(os.path.splitext(file)[0]) for file in filenames])
            torch.save(attention_probs, save_attention_path + f'{max_num + 1}.pth')

        x = torch.sum(attention_probs * x, dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.embed_size) + ' -> ' + str(self.embed_size) + ')'


class value_pooling(nn.Module):
    def __init__(self, embed_size, units=None, use_additive_bias=False, use_attention_bias=False):
        super(value_pooling, self).__init__()
        self.embed_size = embed_size
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.units = units if units else embed_size
        self.U = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.V = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.W = nn.Parameter(torch.Tensor(self.units, self.embed_size))

        nn.init.trunc_normal_(self.U, std=0.01)
        nn.init.trunc_normal_(self.V, std=0.01)
        nn.init.trunc_normal_(self.W, std=0.01)

        if self.use_additive_bias:
            self.b1 = nn.Parameter(torch.Tensor(self.units))
            nn.init.trunc_normal_(self.b1, std=0.01)
        if self.use_attention_bias:
            self.b2 = nn.Parameter(torch.Tensor(self.embed_size))
            nn.init.trunc_normal_(self.b2, std=0.01)

    def forward(self, x, mask=None, save_attention_path=None):
        # (B, Len, Embed) x (Embed, Units) = (B, Len, Units)
        UV = self.U + self.V            # (Embed, Units)  临时张量，不改变参数本身
        h = torch.matmul(x, UV)         # (B, Len, Units)

        if self.use_additive_bias:
            h = torch.tanh(h + self.b1) # b1 broadcast 到 (B, Len, Units)
        else:
            h = torch.tanh(h)

        # (B, Len, Units) x (Units, Embed) = (B, Len, Embed)
        if self.use_attention_bias:
            e = torch.matmul(h, self.W) + self.b2
        else:
            e = torch.matmul(h, self.W)

        if mask is not None:
            attention_probs = nn.Softmax(dim=1)(e + torch.unsqueeze((1.0 - mask) * -10000, dim=-1))
        else:
            attention_probs = nn.Softmax(dim=1)(e)

        if save_attention_path is not None:
            filenames = os.listdir(save_attention_path)
            if len(filenames) == 0:
                max_num = 0
            else:
                max_num = max([int(os.path.splitext(file)[0]) for file in filenames])
            torch.save(attention_probs, save_attention_path + f'{max_num + 1}.pth')

        x = torch.sum(attention_probs * x, dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.embed_size) + ' -> ' + str(self.embed_size) + ')'


class attention_pooling(nn.Module):
    def __init__(self, embed_size, units=256, use_additive_bias=False, dropout_p=0.0, temperature=1.0):
        super(attention_pooling, self).__init__()
        self.embed_size = embed_size
        self.units = units
        self.use_additive_bias = use_additive_bias
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.temperature = temperature

        self.U = nn.Parameter(torch.empty(embed_size, units))
        self.V = nn.Parameter(torch.empty(embed_size, units))
        self.w_token = nn.Parameter(torch.empty(units, 1))

        nn.init.trunc_normal_(self.U, std=0.01)
        nn.init.trunc_normal_(self.V, std=0.01)
        nn.init.trunc_normal_(self.w_token, std=0.01)

        if use_additive_bias:
            self.b1 = nn.Parameter(torch.empty(units))
            nn.init.trunc_normal_(self.b1, std=0.01)

        self.b_token = nn.Parameter(torch.zeros(1))  # 可选：标量 bias

    def forward(self, x, mask=None, save_attention_path=None):
        # x: (B, L, E), mask: (B, L) with 1 for valid, 0 for pad
        q = x @ self.U                      # (B, L, units)
        k = x @ self.V                      # (B, L, units)
        h = q + k
        if self.use_additive_bias:
            h = h + self.b1
        h = torch.tanh(h)                   # (B, L, units)

        s = (h @ self.w_token) + self.b_token  # (B, L, 1)
        s = s / self.temperature

        if mask is not None:
            s = s.masked_fill(mask.unsqueeze(-1) == 0, -1e4)

        alpha = torch.softmax(s, dim=1)     # (B, L, 1)

        if self.dropout is not None:
            alpha = self.dropout(alpha)

        if save_attention_path is not None:
            # 这里保存 alpha 即可，后续可视化也更干净
            torch.save(alpha.detach().cpu(), save_attention_path)

        y = torch.sum(alpha * x, dim=1)     # (B, E)
        return y


class ResidueFeaturePooling(nn.Module):
    def __init__(self, embed_size, units=256, temperature=1.0, residual_init=0.1):
        super().__init__()
        self.temperature = temperature

        self.W = nn.Linear(embed_size, units, bias=True)
        self.v = nn.Linear(units, 1, bias=True)
        nn.init.trunc_normal_(self.W.weight, std=0.01)
        nn.init.zeros_(self.W.bias)
        nn.init.trunc_normal_(self.v.weight, std=0.01)
        nn.init.zeros_(self.v.bias)

        self.beta0 = nn.Parameter(torch.zeros(embed_size))  # global static feature gate
        self.ln_attn = nn.LayerNorm(embed_size)
        self.ln_base = nn.LayerNorm(embed_size)
        self.gamma = nn.Parameter(torch.tensor(float(residual_init)))

    @staticmethod
    def _masked_mean(x, mask):
        m = mask.unsqueeze(-1).to(dtype=x.dtype)            # (B,L,1)
        denom = m.sum(dim=1).clamp(min=1.0)                 # (B,1)
        return (x * m).sum(dim=1) / denom                   # (B,E)

    def forward(self, x, mask=None, save_attention_path=None):
        # x: (B,L,E), mask: (B,L) with 1 valid, 0 pad

        h = torch.tanh(self.W(x))                           # (B,L,U)
        s = self.v(h).squeeze(-1) / self.temperature        # (B,L)

        if mask is not None:
            s = s.masked_fill(mask == 0, torch.finfo(s.dtype).min)

        alpha = F.softmax(s, dim=1)                         # (B,L)

        if save_attention_path is not None:
            torch.save(alpha.detach().cpu(), save_attention_path)

        c_attn = (x * alpha.unsqueeze(-1)).sum(dim=1)       # (B,E)
        c_attn = self.ln_attn(c_attn)
        c_attn = c_attn * torch.sigmoid(self.beta0).unsqueeze(0)

        c0 = x.mean(dim=1) if mask is None else self._masked_mean(x, mask)
        c0 = self.ln_base(c0)

        y = c0 + self.gamma * c_attn                        # (B,E)

        return y