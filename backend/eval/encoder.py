"""Query encoder for all-MiniLM-L6-v2.

Reproduces the SentenceTransformer pipeline declared in the model's
modules.json (Transformer -> mean Pooling -> L2 Normalize) directly on
transformers, so evaluation does not depend on sentence-transformers being
installed and matches the vectors already stored in the FAISS index.
"""
from __future__ import annotations

import numpy as np


class QueryEncoder:
    def __init__(self, model_path: str, max_seq_length: int = 256, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.max_seq_length = max_seq_length
        self.device = device

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_seq_length, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state          # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()    # (B, T, 1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy())
        return np.vstack(out).astype(np.float32)
