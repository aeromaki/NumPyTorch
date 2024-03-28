import sys
import warnings
sys.setrecursionlimit(5000)
warnings.filterwarnings("ignore")

from typing import Optional, List, Tuple

import numpytorch as npt
from numpytorch import Tensor, nn, optim
import numpy as np


n_vocab = 30522
d_model = 32
d_k = 16
d_v = 16
n_head = 2
n_layer = 1
max_len = 1024
batch_size = 2


class MyTransformerModel(nn.Module):
    def __init__(self) -> None:
        self.transformer = nn.Transformer(n_vocab, d_model, d_k, d_v, n_head, n_layer, max_len)
        self.head = nn.Linear(d_model, n_vocab)

    def forward(
        self,
        e_ids: Tensor,
        d_ids: Tensor,
        mask_e: Optional[Tensor] = None,
        mask_d: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        d_last_hidden_state = self.transformer(e_ids, d_ids, mask_e, mask_d, mask_tgt)
        logits = self.head(d_last_hidden_state)
        return logits


from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


class Tokenizer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        input_ids, _, attn_mask = self.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length"
        ).values()
        return npt.tensor(input_ids), npt.tensor(attn_mask)


tokenizer = Tokenizer()
model = MyTransformerModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=6e-05)
dataset = load_dataset("opus_books", "en-fr")


print("train start")
for i in tqdm(range(127085//batch_size)):
    x_texts, y_texts = [*zip(*map(lambda x: [*x.values()], dataset["train"]["translation"][i*batch_size:(i+1)*batch_size]))]
    x_ids, x_mask = tokenizer(x_texts)
    y_ids, y_mask = tokenizer(y_texts)

    optimizer.zero_grad()

    logits = model(x_ids, y_ids, x_mask, y_mask, nn.Transformer.create_tgt_mask(x_ids.shape[-1]))

    logit_mask = npt.ones(*logits.shape)
    logit_mask[y_mask == 0] = -npt.inf
    logits = logits * logit_mask

    loss = criterion(logits, y_ids)
    loss.backward()
    print(loss.item())

    optimizer.step()