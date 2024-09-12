import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from gensim.downloader import load
from torch.types import Device
from tqdm import tqdm


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cosine_similarity(x[..., None], y.T, dim=-2)


@dataclass
class Embeddings:
    vectors: torch.Tensor
    i2w: list[str]
    w2i: dict[str, int]

    def most_similar_ix(self, vector: torch.Tensor, top_n: int = 10, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is None:
            similarities = cosine_similarity(vector, self.vectors)
        else:
            splits = self.vectors.split(batch_size, dim=0)
            similarities = torch.cat([cosine_similarity(vector, split).cpu() for split in splits], dim=-1)
        return similarities.topk(top_n).indices

    def __len__(self):
        return len(self.i2w)

    def to(self, device: Device):
        self.vectors = self.vectors.to(device)
        return self

    @staticmethod
    def from_file(path: Path | str, max_words: int | None = None):
        embeddings, i2w, w2i = [], [], {}
        with open(path, "r", newline="\n", errors="ignore") as file:
            next(file)
            for line in tqdm(file, desc=f"Loading embeddings from {path}", total=max_words, leave=False):
                word, vector = line.rstrip().split(" ", 1)
                vector = np.fromstring(vector, sep=" ")
                embeddings.append(vector)
                w2i[word] = len(w2i)
                i2w.append(word)
                if max_words and len(w2i) == max_words:
                    break
        return Embeddings(torch.Tensor(np.stack(embeddings)), i2w, w2i)

    @staticmethod
    def from_gensim(name: str):
        model = load(name)
        return Embeddings(torch.Tensor(model.vectors), model.index_to_key, model.key_to_index)

    def normalize(self):
        self.vectors = torch.nn.functional.normalize(self.vectors)

    def restrict(self, words: Sequence[str], reset_index: bool = True):
        i2w = [w for w, _ in itertools.groupby(words)]
        ix = torch.tensor([self.w2i[word] for word in i2w if word in self.w2i])
        vectors = self.vectors[ix]
        if reset_index:
            w2i = {w: i for i, w in enumerate(i2w)}
            return Embeddings(vectors, i2w, w2i)

        w2i = {w: i for w in i2w if (i := self.w2i.get(w)) is not None}
        return RestrictedEmbeddings(vectors, i2w, w2i, ix)


@dataclass
class RestrictedEmbeddings(Embeddings):
    _ix: torch.Tensor

    def most_similar_ix(self, vector: torch.Tensor, top_n: int = 10, batch_size: int | None = None) -> torch.Tensor:
        return self._ix[super().most_similar_ix(vector, top_n, batch_size)]

    def to(self, device: Device):
        self._ix = self._ix.to(device)
        return super().to(device)
