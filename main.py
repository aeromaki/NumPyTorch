import gzip
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from sklearn.metrics import f1_score

import numpytorch as npt
from numpytorch import tensor, nn, optim
from assignment import MNISTClassificationModel


def get_mnist() -> tuple[ndarray, ndarray, ndarray, ndarray]:
    path_x = 'data/train-images-idx3-ubyte.gz'
    path_y = 'data/train-labels-idx1-ubyte.gz'

    with gzip.open(path_x, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    x = images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.

    with gzip.open(path_y, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    y = labels

    x_train, x_val = x[:50000], x[50000:]
    y_train, y_val = y[:50000], y[50000:]
    return (x_train, y_train, x_val, y_val)


def val(model: nn.Module, x_val: ndarray, y_val: ndarray) -> tuple[float, float]:
    preds = []
    for i in range(0, len(x_val), 32):
        x = tensor(x_val[i:i+32])
        preds += model(x).arr.argmax(-1).tolist()
    macro = f1_score(y_val, preds, average="macro")
    micro = f1_score(y_val, preds, average="micro")
    return macro, micro


if __name__ == '__main__':
    x_train, y_train, x_val, y_val = get_mnist()

    ### edits allowed here ###
    n_batch = 32
    n_iter = 50000
    n_print = 100
    n_val = 2000
    lr = 1e-03
    ##########################

    model = MNISTClassificationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    buf = 0
    for i in tqdm(range(1, n_iter+1)):
        idx = np.random.permutation(50000)[:n_batch]
        x, y = tensor(x_train[idx]), tensor(y_train[idx])

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        if np.isnan(loss.item()):
            break

        loss.backward()
        optimizer.step()

        buf += loss.item()
        if i % n_print == 0:
            print(buf / n_print)
            buf = 0

        if i % n_val == 0:
            macro, micro = val(model, x_val, y_val)
            print(f"macro: {macro:.9f} micro: {micro:.9f}")

    macro, micro = val(model, x_val, y_val)
    print(f"\nfinal score\nmacro: {macro:.9f} micro: {micro:.9f}")