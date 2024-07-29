import numpy as np

from typing import Optional

from .base import Module, Parameter, Linear
from .nodes import Sequential, ModuleList
from .layernorm import LayerNorm
from .functions import ReLU
from numpytorch.tensor import Tensor
from numpytorch.functions import reshape, transpose, ones, unsqueeze, tensor, softmax


class _AttentionProjector(Module):
    def __init__(
        self,
        d_model: int,
        d_proj: int,
        n_head: int
    ) -> None:
        self.proj = Linear(d_model, d_proj*n_head)
        self.d_proj = d_proj
        self.n_head = n_head

    def forward(self, x: Tensor) -> Tensor:
        p = self.proj(x)
        p = reshape(p, (*p.shape[:-2], self.n_head, p.shape[-2], self.d_proj))
        return p

class MultiHeadAttention(Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int
    ) -> None:
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.Wq = _AttentionProjector(d_model, d_k, n_head)
        self.Wk = _AttentionProjector(d_model, d_k, n_head)
        self.Wv = _AttentionProjector(d_model, d_v, n_head)
        self.Wo = Linear(d_v*n_head, d_model)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask_q: Optional[Tensor] = None,
        mask_k: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        q_mh = self.Wq(q)
        k_mh = self.Wk(k)
        v_mh = self.Wv(v)
        mask_q_mh = self._generate_mask(mask_q)
        mask_k_mh = self._generate_mask(mask_k)

        q_mh = q_mh * mask_q_mh
        k_mh = k_mh * mask_k_mh
        inn = q_mh @ transpose(k_mh)

        if mask_tgt is not None:
            mask_tgt_ = ones(*mask_tgt.shape)
            mask_tgt_[mask_tgt == 0] = -np.inf
            inn = inn * mask_tgt_

        att = softmax(inn / self.d_k ** 0.5) @ v_mh

        res = self._reshape_v(att)
        res = self.Wo(res)
        return res

    @staticmethod
    def _generate_mask(mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            mask_mh = ones(*mask.shape)
            mask_mh[mask == 0] = -np.inf
            mask_mh = unsqueeze(unsqueeze(mask_mh, 1), mask_mh.ndim+1)
        else:
            mask_mh = None
        return mask_mh

    @staticmethod
    def _reshape_v(output: Tensor) -> Tensor:
        res = transpose(output, (-3, -2))
        res = reshape(res, (*res.shape[:-2], -1))
        return res

class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        n_head: int
    ) -> None:
        self.att_self = MultiHeadAttention(d_model, d_k, d_k, n_head)
        self.ff = Sequential(
            Linear(d_model, d_model*4),
            ReLU(),
            Linear(d_model*4, d_model)
        )
        self.layernorm0 = LayerNorm()
        self.layernorm1 = LayerNorm()

    def forward(
        self,
        q: Tensor,
        mask_q: Optional[Tensor] = None
    ):
        h_self = self.att_self(q, q, q, mask_q, mask_q)
        res_self = self.layernorm0(q + h_self)

        h_ff = self.ff(res_self)
        res = self.layernorm1(res_self + h_ff)

        return res

class TransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int
    ):
        self.att_self = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.att_cross = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.ff = Sequential(
            Linear(d_model, d_model*4),
            ReLU(),
            Linear(d_model*4, d_model)
        )
        self.layernorm0 = LayerNorm()
        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        mask_q: Optional[Tensor] = None,
        mask_k: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ):
        h_self = self.att_self(q, q, q, mask_q, mask_q, mask_tgt)
        res_self = self.layernorm0(q + h_self)

        h_cross = self.att_cross(res_self, k, k, mask_q, mask_k)
        res_cross = self.layernorm1(res_self + h_cross)

        h_ff = self.ff(res_cross)
        res = self.layernorm2(res_cross + h_ff)

        return res

class TransformerEncoder(Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        n_head: int,
        n_layer: int
    ) -> None:
        self.layers = ModuleList([
            TransformerEncoderLayer(d_model, d_k, n_head) for _ in range(n_layer)
        ])

    def forward(
        self,
        q: Tensor,
        mask_q: Optional[Tensor] = None,
    ) -> Tensor:
        h = q
        for layer in self.layers:
            h = layer(h, mask_q)
        return h

class TransformerDecoder(Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int,
        n_layer: int
    ) -> None:
        self.layers = ModuleList([
            TransformerDecoderLayer(d_model, d_k, d_v, n_head) for _ in range(n_layer)
        ])

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        mask_q: Optional[Tensor] = None,
        mask_k: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        h = q
        for layer in self.layers:
            h = layer(h, k, mask_q, mask_k, mask_tgt)
        return h

class Embedding(Module):
    def __init__(
        self,
        n_vocab: int,
        d_model: int
    ) -> None:
        self.embedding = Parameter.new(n_vocab, d_model)

    def forward(self, seq: Tensor) -> Tensor:
        return self.embedding[seq]

class Transformer(Module):
    def __init__(
        self,
        n_vocab: int,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int,
        n_layer: int,
        max_len: int
    ):
        self.embedding = Embedding(n_vocab, d_model)
        self.max_len = max_len
        self.positional_embedding = Parameter.new(max_len, d_model)
        self.encoder = TransformerEncoder(d_model, d_k, n_head, n_layer)
        self.decoder = TransformerDecoder(d_model, d_k, d_v, n_head, n_layer)

    def forward(
        self,
        e_ids: Tensor,
        d_ids: Tensor,
        mask_e: Optional[Tensor] = None,
        mask_d: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        e = self.embed(e_ids)
        d = self.embed(d_ids)
        e_last_hidden_state = self.encoder(e, mask_e)
        d_last_hidden_state = self.decoder(d, e_last_hidden_state, mask_d, mask_e, mask_tgt)
        return d_last_hidden_state

    def embed(self, x: Tensor) -> Tensor:
        assert x.shape[-1] <= self.max_len
        d = self.embedding(x)
        d = d + self.positional_embedding[:x.shape[-1], :]
        return d

    @staticmethod
    def create_tgt_mask(tgt_len: int) -> Tensor:
        mask = tensor(np.triu(np.ones((tgt_len, tgt_len))))
        return mask