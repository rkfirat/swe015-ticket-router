"""Train the four-model ladder on the Bitext customer-support data (27 intents, 11 teams).

    python train.py                 # everything
    python train.py tfidf ffn       # selected stages: tfidf ffn bert distill
"""
from __future__ import annotations

import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.common import MODELS, SEED, get_corrector, load_data, normalize
from src.models import FFN, average_vectors

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def data():
    splits, names, oos, _ = load_data()
    get_corrector()  # builds models/vocab.json from the training split only
    split = {k: ([normalize(t) for t in splits[k][0]], np.array(splits[k][1]))
             for k in ("train", "validation")}
    # Checkpoints are picked on in-scope validation accuracy; OOS recall is tuned later with the threshold.
    xva, yva = split["validation"]
    keep = yva != oos
    split["validation"] = ([t for t, k in zip(xva, keep) if k], yva[keep])
    return split, oos + 1  # 27 intents + the "oos" class learned from CLINC negatives


# --- 1. tf-idf + logistic regression ------------------------------------------

def train_tfidf(split, n_classes):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion

    (xtr, ytr), (xva, yva) = split["train"], split["validation"]
    # Word 1-2 grams plus character 2-5 grams: typos ("acount") still share most pieces.
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=20, max_iter=2000)
    clf.fit(vec.fit_transform(xtr), ytr)
    log(f"tf-idf val acc {clf.score(vec.transform(xva), yva):.4f}")
    with open(MODELS / "tfidf_lr.pkl", "wb") as f:
        pickle.dump((vec, clf), f)


# --- 2. fastText + feedforward network (SGD + backprop) ---------------------

def train_ffn(split, n_classes, epochs=60, hidden=256):
    from gensim.models import FastText

    (xtr, ytr), (xva, yva) = split["train"], split["validation"]
    w2v = FastText([t.split() for t in xtr], vector_size=200, window=5, min_count=1,
                   sg=1, epochs=40, min_n=3, max_n=5, bucket=100_000, seed=SEED, workers=1)
    w2v.wv.save(str(MODELS / "fasttext.kv"))

    Xtr = torch.from_numpy(average_vectors(w2v.wv, xtr))
    Xva = torch.from_numpy(average_vectors(w2v.wv, xva))
    Ytr, Yva = torch.from_numpy(ytr), torch.from_numpy(yva)

    net = FFN(Xtr.shape[1], hidden, n_classes)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    best, best_state = 0.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(net(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xva).argmax(-1) == Yva).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    log(f"ffn best val acc {best:.4f}")
    torch.save({"state": best_state, "dim": Xtr.shape[1], "hidden": hidden,
                "n_classes": n_classes}, MODELS / "ffn.pt")


# --- 3. BERT fine-tuning ------------------------------------------------------

def batches(texts, labels, bs, shuffle):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield [texts[k] for k in j], torch.from_numpy(labels[j]), j


def evaluate_hf(model, tok, texts, labels):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb, _ in batches(texts, labels, 128, False):
            enc = tok(xb, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            correct += (model(**enc).logits.argmax(-1).cpu() == yb).sum().item()
    return correct / len(texts)


def fine_tune(model, tok, split, out_dir, epochs, lr, bs=32, teacher_logits=None, T=2.0, alpha=0.5):
    from transformers import get_linear_schedule_with_warmup

    (xtr, ytr), (xva, yva) = split["train"], split["validation"]
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(xtr) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb, j in batches(xtr, ytr, bs, True):
            enc = tok(xb, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            loss = F.cross_entropy(logits, yb.to(DEVICE))
            if teacher_logits is not None:  # Hinton distillation: soft targets at temperature T
                soft = F.log_softmax(teacher_logits[j].to(DEVICE) / T, dim=-1)
                kd = F.kl_div(F.log_softmax(logits / T, dim=-1), soft, log_target=True,
                              reduction="batchmean") * T * T
                loss = alpha * kd + (1 - alpha) * loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = evaluate_hf(model, tok, xva, yva)
        log(f"{out_dir.name} epoch {ep + 1}/{epochs} val acc {acc:.4f}")
        if acc > best:
            best = acc
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    log(f"{out_dir.name} best val acc {best:.4f}")


def train_bert(split, n_classes):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_classes)
    fine_tune(model, tok, split, MODELS / "bert", epochs=3, lr=4e-5)


def train_distill(split, n_classes):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    teacher_dir = MODELS / "bert"
    tok_t = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir).to(DEVICE).eval()
    xtr, ytr = split["train"]
    logits = []
    with torch.no_grad():
        for xb, _, _ in batches(xtr, ytr, 128, False):
            enc = tok_t(xb, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            logits.append(teacher(**enc).logits.cpu())
    teacher_logits = torch.cat(logits)
    del teacher

    name = "google/bert_uncased_L-4_H-256_A-4"  # BERT-mini: 4 layers, 256 hidden
    tok = AutoTokenizer.from_pretrained(name)
    student = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_classes)
    fine_tune(student, tok, split, MODELS / "student", epochs=15, lr=2e-4, bs=64,
              teacher_logits=teacher_logits)


STAGES = {"tfidf": train_tfidf, "ffn": train_ffn, "bert": train_bert, "distill": train_distill}

if __name__ == "__main__":
    seed_all()
    split, n_classes = data()
    for stage in sys.argv[1:] or list(STAGES):
        log(f"stage {stage}")
        STAGES[stage](split, n_classes)
