from __future__ import annotations
from typing import Any, Callable, List
from .tensor import Tensor, Value
from .functions import *
from . import inf


class Parameter(Tensor):
    """
    To manage the tensors used as parameters in the model separately,
    we created this class that inherits from Tensor.
    """
    def __init__(self, x: Tensor) -> None:
        super().__init__(arr=x, requires_grad=True)

    def _init_weight(*args: int) -> Tensor:
        # He Uniform Initialization
        u = (6 / args[0])**0.5
        return tensor(np.random.uniform(-u, u, size=args))

    @staticmethod
    def new(*args: int) -> Parameter:
        return Parameter(Parameter._init_weight(*args))

    @staticmethod
    def new_scalar() -> Parameter:
        return Parameter(rand())


class Module:
    """
    A class for conveniently managing each layer, module, or model of a DNN.
    If you want to create a new layer, you can create a subclass that inherits
    from Module and just implement the forward method.
    """
    def _forward_unimplemented(*args, **kwargs) -> None:
        raise Exception("forward not implemented")
    forward: Callable[..., Any] = _forward_unimplemented

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Parameter]:
        """
        In order to optimize a model during training, the values of the parameters inside
        the model must be constantly updated. This is done through the optimizer in optim.py,
        which requires a list of all the parameters (Parameter) a model (or module) has.
        If a Module contains other Modules as attributes, it will also return the parameters
        of those Modules.
        """
        params: List[Parameter] = []
        for v in self.__dict__.values():
            if isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, Parameter):
                params.append(v)
        return params


class Linear(Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        self.w = Parameter.new(d_in, d_out)
        self.b: Value = Parameter(zeros(d_out)) if bias else 0

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b

class Sequential(Module):
    """
    It is often the case that multiple layers need to be applied in succession, each taking
    a single tensor as input and returning a single tensor (e.g. CNN). It's tedious to assign
    each layer an attribute for this process and apply each one directly in the forward, so we
    can wrap it in a simple Module.
    """
    def __init__(self, *args) -> None:
        for i, module in enumerate(args):
            setattr(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.__dict__.values():
            x = layer(x)
        return x

class ModuleList(Module, list):
    def __init__(self, modules: List[Module]) -> None:
        super().__init__(modules)
        for i, module in enumerate(modules):
            setattr(self, str(i), module)

class ReLU(Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return relu(x)

class Sigmoid(Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return sigmoid(x)

class Tanh(Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return tanh(x)

class CrossEntropyLoss(Module):
    @staticmethod
    def forward(logits: Tensor, q: Tensor) -> Tensor:
        if logits.shape != q.shape:
            q = one_hot(q, logits.shape[-1])
        log_p = logits - log(sum(exp(logits), -1, keepdims=True))
        ce = -sum(q * log_p, -1)
        return mean(ce)

class LayerNorm(Module):
    def __init__(
        self,
        eps: float = 1e-05
    ) -> None:
        self.eps = eps
        self.gamma = Parameter.new_scalar()
        self.beta = Parameter.new_scalar()

    def forward(self, x: Tensor) -> Tensor:
        return (x - mean(x, -1, keepdims=True)) / (var(x, -1) + self.eps) ** 0.5 * self.gamma + self.beta

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
            mask_tgt_[mask_tgt == 0] = -inf
            inn = inn * mask_tgt_

        att = softmax(inn / self.d_k ** 0.5) @ v_mh

        res = self._reshape_v(att)
        res = self.Wo(res)
        return res

    @staticmethod
    def _generate_mask(mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            mask_mh = ones(*mask.shape)
            mask_mh[mask == 0] = -inf
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