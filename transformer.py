import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Absoluteposition(nn.Module):
    def __init__(self, n_embd, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, n_embd)

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, n_embd)

        self.register_buffer("pe", pe)  # 不会被训练

    def forward(self, x):
        B, T, C = x.shape
        return x + self.pe[:, :T, :]





class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class MultiHeadSelfAttention(nn.Module):
    """
    Return:
      y:   (B, T, C)
      att: (B, n_head, T, T)
    """
    def __init__(self, n_embd, n_head, causal: bool = False, use_alibi: bool = False):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal
        self.use_alibi = use_alibi

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        if use_alibi:
            slopes = self._get_alibi_slopes(n_head)
            self.register_buffer(
                "alibi_slopes",
                torch.tensor(slopes).view(1, n_head, 1, 1)
            )

    def _get_alibi_slopes(self, n_head):
        import math
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_head).is_integer():
            return get_slopes_power_of_2(n_head)
        else:
            closest = 2 ** math.floor(math.log2(n_head))
            return (
                get_slopes_power_of_2(closest)
                + self._get_alibi_slopes(2 * closest)[0::2][: n_head - closest]
            )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 🔥 ALiBi bias
        if self.use_alibi:
            pos = torch.arange(T, device=x.device)
            rel_dist = pos[None, :] - pos[:, None]   # (T, T)
            rel_dist = rel_dist.clamp(min=0)         # causal distance
            rel_dist = rel_dist.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            scores = scores - self.alibi_slopes * rel_dist

        # causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        att = F.softmax(scores, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return y, att





class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_hidden, causal: bool = False):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, causal=causal)
        self.ln2 = LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, n_hidden)

    def forward(self, x):
        att_out, att = self.attn(self.ln1(x))
        x = x + att_out
        x = x + self.ff(self.ln2(x))
        return x, att



class TransformerClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 block_size,
                 n_embd,
                 n_head,
                 n_layer,
                 n_hidden,
                 n_output):
        super().__init__()
        self.block_size = block_size
        self.n_head = n_head

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_enc = Absoluteposition(n_embd, max_len=block_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, n_hidden) for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(n_embd)

        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"T={T} exceeds block_size={self.block_size}")

        idx = idx.long()

        # token embedding: (B, T, C)
        tok_emb = self.token_embedding(idx)
        x = self.pos_enc(tok_emb)   

        attn_maps = []
        for block in self.blocks:
            x, att = block(x)  # att: (B, n_head, T, T)

            # flatten heads into list elements: each (B,T,T) to match utilities.py
            for h in range(att.size(1)):
                attn_maps.append(att[:, h, :, :])

        x = self.ln_f(x)       # (B, T, C)
        x = x.mean(dim=1)      # (B, C)
        logits = self.classifier(x)  # (B, n_output)

        return logits, attn_maps
    
class TransformerDecoderLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 block_size,
                 n_embd,
                 n_head,
                 n_layer,
                 n_hidden=100,
                 return_attn_maps=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.return_attn_maps = return_attn_maps

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_enc = Absoluteposition(n_embd, max_len=block_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, n_hidden, causal=True) for _ in range(n_layer)]
        )

        self.ln_f = LayerNorm(n_embd)

        # lm head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"T={T} exceeds block_size={self.block_size}")

        idx = idx.long()
        x = self.token_embedding(idx)   # (B, T, C)
        x = self.pos_enc(x)             #positional encoding

        attn_maps = [] if self.return_attn_maps else None

        for block in self.blocks:
            x, att = block(x)  # att: (B, n_head, T, T)
            if self.return_attn_maps:
                for h in range(att.size(1)):
                    attn_maps.append(att[:, h, :, :])  # (B, T, T)

        x = self.ln_f(x)                 # (B, T, C)
        logits = self.lm_head(x)         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            targets = targets.long()
            loss = F.cross_entropy(
                logits.view(B * T, self.vocab_size),
                targets.view(B * T)
            )

        if self.return_attn_maps:
            return loss, attn_maps
        return logits, loss

class TransformerDecoderAlibi(nn.Module):
    def __init__(self,
                 vocab_size,
                 block_size,
                 n_embd,
                 n_head,
                 n_layer,
                 n_hidden=100,
                 return_attn_maps=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.return_attn_maps = return_attn_maps

        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, n_hidden, causal=True)
             for _ in range(n_layer)]
        )

        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"T={T} exceeds block_size={self.block_size}")

        idx = idx.long()
        x = self.token_embedding(idx)   # 🔥 no positional embedding

        attn_maps = [] if self.return_attn_maps else None

        for block in self.blocks:
            x, att = block(x)
            if self.return_attn_maps:
                for h in range(att.size(1)):
                    attn_maps.append(att[:, h, :, :])

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            targets = targets.long()
            loss = F.cross_entropy(
                logits.view(B * T, self.vocab_size),
                targets.view(B * T)
            )

        if self.return_attn_maps:
            return loss, attn_maps

        return logits, loss
