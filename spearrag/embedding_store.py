from copy import deepcopy
from spearrag.utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename: str, batch_size: int, namespace: str):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace

        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}

        self._load()

    def _load(self):
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: i for i, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = dict(zip(self.hash_ids, self.texts))
            self.text_to_hash_id = dict(zip(self.texts, self.hash_ids))
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")

    def insert_text(self, text_list: list):
        nodes = {
            compute_mdhash_id(t, prefix=self.namespace + "-"): t
            for t in text_list
        }
        existing = set(self.hash_ids)
        missing = [(h, t) for h, t in nodes.items() if h not in existing]
        if not missing:
            return
        ids, texts = zip(*missing)
        embs = self.embedding_model.encode(
            list(texts), normalize_embeddings=True,
            show_progress_bar=False, batch_size=self.batch_size
        )
        self.hash_ids.extend(ids)
        self.texts.extend(texts)
        self.embeddings.extend(embs)
        self.hash_id_to_idx = {h: i for i, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = dict(zip(self.hash_ids, self.texts))
        self.text_to_hash_id = dict(zip(self.texts, self.hash_ids))
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings,
        }).to_parquet(self.db_filename, index=False)

    def get_hash_id_to_text(self):
        return deepcopy(self.hash_id_to_text)

    def get_embeddings(self, hash_ids: list) -> np.ndarray:
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        return np.array(self.embeddings)[indices]
