"""The four models of the ladder behind one interface: predict_proba(list[str]) -> (n, 151)."""
from __future__ import annotations

import pickle

import numpy as np
import torch
from torch import nn

from .common import MODELS, normalize

torch.backends.quantized.engine = "qnnpack"  # ARM CPU backend for INT8 kernels


class Model:
    name: str
    precision: str
    files: list

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def size_mb(self) -> float:
        return sum(f.stat().st_size for f in self.files) / 1e6


# --- 1. Classical baseline: tf-idf + logistic regression ---------------------

class TfidfLR(Model):
    name, precision = "tf-idf + LogReg", "FP64"
    path = MODELS / "tfidf_lr.pkl"

    def __init__(self):
        with open(self.path, "rb") as f:
            self.vec, self.clf = pickle.load(f)
        self.files = [self.path]

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform([normalize(t) for t in texts]))


# --- 2. Neural network trained from scratch: averaged word2vec -> FFN --------

class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def average_vectors(kv, texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in t.split() if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


class W2VFFN(Model):
    name, precision = "word2vec + FFN", "FP32"
    w2v_path = MODELS / "word2vec.kv"
    ffn_path = MODELS / "ffn.pt"

    def __init__(self):
        from gensim.models import KeyedVectors

        self.kv = KeyedVectors.load(str(self.w2v_path))
        ckpt = torch.load(self.ffn_path, weights_only=True)
        self.net = FFN(ckpt["dim"], ckpt["hidden"], ckpt["n_classes"])
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.files = [self.ffn_path, *MODELS.glob("word2vec.kv*")]

    @torch.no_grad()
    def predict_proba(self, texts):
        x = torch.from_numpy(average_vectors(self.kv, [normalize(t) for t in texts]))
        return torch.softmax(self.net(x), dim=-1).numpy()


# --- 3/4. Transformers: fine-tuned BERT, distilled + INT8 student -------------

class HFClassifier(Model):
    def __init__(self, path, name, precision, quantize=False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name, self.precision = name, precision
        self.tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        if quantize:
            q = torch.ao.quantization
            # Linear layers: dynamic INT8 (weights int8, activations quantized per batch at run time).
            # Embeddings: 8-bit weight-only, otherwise the 30k-token table dominates the size.
            model = q.quantize_dynamic(model, {nn.Linear: q.default_dynamic_qconfig,
                                               nn.Embedding: q.float_qparams_weight_only_qconfig})
            qpath = path / "int8_state.pt"
            torch.save(model.state_dict(), qpath)
            self.files = [qpath]
        else:
            self.files = list(path.glob("*.safetensors"))
        self.model = model

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [normalize(t) for t in texts[i:i + batch_size]]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, dim=-1).numpy())
        return np.concatenate(out)


def bert():
    return HFClassifier(MODELS / "bert", "BERT-base fine-tuned", "FP32")


def student_int8():
    return HFClassifier(MODELS / "student", "BERT-mini distilled", "INT8 (dynamic)", quantize=True)


def bert_int8():
    return HFClassifier(MODELS / "bert", "BERT-base fine-tuned", "INT8 (dynamic)", quantize=True)


def student_fp32():
    return HFClassifier(MODELS / "student", "BERT-mini distilled", "FP32")


def load_all(with_ablations: bool = False) -> list[Model]:
    """The four-model ladder; ablations add BERT-base INT8 and the student before quantization."""
    models = [TfidfLR(), W2VFFN(), bert()]
    if with_ablations:
        models += [bert_int8(), student_fp32()]
    return models + [student_int8()]
