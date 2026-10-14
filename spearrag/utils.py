"""Small helpers shared by the retriever, the reranker and the evaluation script."""
import re
import string
from hashlib import md5


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def normalize_answer(s) -> str:
    """SQuAD-style answer normalisation used by EM / F1 / Contain-Acc."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
