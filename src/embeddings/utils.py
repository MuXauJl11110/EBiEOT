import itertools
from pathlib import Path
from typing import Sequence

import torch
from torch.types import Device

from src.embeddings.base import Embeddings, RestrictedEmbeddings


def restrict(embeddings: Embeddings, words: Sequence[str], reset_index: bool = True) -> RestrictedEmbeddings:
    i2w = [w for w, _ in itertools.groupby(words)]
    ix = torch.tensor(
        [embeddings.w2i[word] for word in i2w if word in embeddings.w2i], device=embeddings.vectors.device
    )
    vectors = embeddings.vectors[ix]
    if reset_index:
        w2i = {w: i for i, w in enumerate(i2w)}
        return Embeddings(vectors, i2w, w2i)

    w2i = {w: i for w in i2w if (i := embeddings.w2i.get(w)) is not None}
    return RestrictedEmbeddings(vectors, i2w, w2i, ix)


def load_dictionary_words(path: str | Path) -> tuple[str, str]:
    source_words, target_words = [], []
    with open(path, "r", newline="\n", errors="ignore") as f:
        for line in f:
            word_src, word_tgt = line.rstrip().split()
            source_words.append(word_src)
            target_words.append(word_tgt)
    return source_words, target_words


def make_dictionary(source_words: list[str], target_words: list[str]) -> dict[str, list[str]]:
    dictionary = {}
    for word_src, word_tgt in zip(source_words, target_words):
        dictionary.setdefault(word_src, []).append(word_tgt)
    return dictionary


def get_vectors_and_labels(
    emb_src: Embeddings, words_src: list[str], emb_tgt: Embeddings, words_tgt: list[str], device: Device
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors_src, indices_src = [], []
    vectors_tgt, indices_tgt = [], []

    for word_src, word_tgt in zip(words_src, words_tgt):
        if word_src in emb_src.w2i and word_tgt in emb_tgt.w2i:
            indices_src.append(emb_src.w2i[word_src])
            vectors_src.append(emb_src.vectors[emb_src.w2i[word_src]])

            indices_tgt.append(emb_tgt.w2i[word_tgt])
            vectors_tgt.append(emb_tgt.vectors[emb_tgt.w2i[word_tgt]])

    return (
        torch.stack(vectors_src, dim=0).to(device),
        torch.tensor(indices_src).to(device),
        torch.stack(vectors_tgt, dim=0).to(device),
        torch.tensor(indices_tgt).to(device),
    )


def foscttm(x: torch.Tensor, y: torch.Tensor) -> float:
    d = torch.cdist(x, y)
    foscttm_x = (d < torch.unsqueeze(torch.diag(d), dim=1)).float().mean(dim=1)
    foscttm_y = (d < torch.unsqueeze(torch.diag(d), dim=0)).float().mean(dim=0)
    fracs = (foscttm_x + foscttm_y) / 2
    return torch.mean(fracs).round(decimals=4).item()
