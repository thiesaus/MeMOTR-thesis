from typing import List, Tuple
import json
import random

def load_predefined_vocab() -> dict:
    vocab_path: str = "data/new_categories.json"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    
    return vocab


def init_guide(labels: list, cls_map: dict, guide_shape: int = 80) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    res = []
    pos_res = []
    labels = [int(_) for _ in labels]
    pos = list(set(labels))
    neg_len = guide_shape - len(pos)
    for label in labels:
        try:
            new_label = cls_map[label]
            res.append([new_label])
            if label in pos:
                pos_res.append([new_label])
                pos.remove(label)
        except:
            raise ValueError(f"Label {label} not found in cls_map")
    
    predefined_vocab = load_predefined_vocab()
    neg_res = []
    neq_idx = list(range(len(predefined_vocab)))
    random.shuffle(neq_idx)
    neq_idx = neq_idx[:neg_len]
    for idx in neq_idx:
        rand_vocab = predefined_vocab[idx]
        new_label = rand_vocab["name"]
        if random.random() < 0.5:
            new_label = rand_vocab["synonyms"][random.randint(0, len(rand_vocab["synonyms"]) - 1)]
        neg_res.append([new_label])

    return res, pos_res, neg_res