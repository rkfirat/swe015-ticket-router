"""Shared paths, data loading and the text normalizer (area B)."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SEED = 42


def load_clinc():
    """CLINC150 'plus' split: 150 intents + out-of-scope, fixed train/val/test."""
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    return ds, names, names.index("oos")


# --- B: regex normalization ------------------------------------------------

# Upper-case product/model codes like "DW-6534" or "RF5000" (so "my 401k" is not one).
MODEL_CODE = re.compile(r"\b([A-Z]{1,4})-?(\d{3,5}[A-Z]?)\b")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
REPEAT = re.compile(r"(\w)\1{2,}")  # "sooooo" -> "soo"
NON_WORD = re.compile(r"[^a-z0-9<>' ]+")
SPACES = re.compile(r"\s+")


def extract_model_codes(text: str) -> list[str]:
    return [f"{a.upper()}-{b.upper()}" for a, b in MODEL_CODE.findall(text)]


def regex_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = URL.sub(" <url> ", text)
    text = EMAIL.sub(" <email> ", text)
    text = MODEL_CODE.sub(" <model> ", text)
    text = text.lower()
    text = REPEAT.sub(r"\1\1", text)
    text = NON_WORD.sub(" ", text)
    return SPACES.sub(" ", text).strip()


# --- B: minimum edit distance spelling correction --------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, two-row dynamic programming."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_english_word(w: str) -> bool:
    """Real English words outside the training vocabulary (e.g. "draining") are left alone."""
    from wordfreq import zipf_frequency

    return zipf_frequency(w, "en") >= 3.0


class SpellCorrector:
    """Replaces unknown words with the closest frequent training word (minimum edit distance)."""

    def __init__(self, vocab: Counter, min_count: int = 2):
        self.freq = {w: c for w, c in vocab.items() if c >= min_count}
        self.by_len: dict[int, list[str]] = {}
        for w in self.freq:
            self.by_len.setdefault(len(w), []).append(w)

    @classmethod
    def from_texts(cls, texts) -> "SpellCorrector":
        return cls(Counter(w for t in texts for w in regex_normalize(t).split()))

    def save(self, path: Path):
        path.write_text(json.dumps(self.freq))

    @classmethod
    def load(cls, path: Path) -> "SpellCorrector":
        return cls(Counter(json.loads(path.read_text())), min_count=0)

    def correct_word(self, w: str) -> str:
        if w in self.freq or len(w) < 4 or not w.isalpha() or is_english_word(w):
            return w
        max_d = 1 if len(w) < 8 else 2
        best, best_key = w, None
        for n in range(len(w) - max_d, len(w) + max_d + 1):
            for cand in self.by_len.get(n, ()):
                d = edit_distance(w, cand)
                if d <= max_d:
                    key = (d, -self.freq[cand])
                    if best_key is None or key < best_key:
                        best, best_key = cand, key
        return best

    def __call__(self, text: str) -> str:
        return " ".join(self.correct_word(w) for w in text.split())


@lru_cache(maxsize=1)
def get_corrector() -> SpellCorrector:
    path = MODELS / "vocab.json"
    if path.exists():
        return SpellCorrector.load(path)
    ds, _, _ = load_clinc()
    sc = SpellCorrector.from_texts(ds["train"]["text"])
    sc.save(path)
    return sc


def normalize(text: str, correct: bool = True) -> str:
    text = regex_normalize(text)
    return get_corrector()(text) if correct else text


# --- B: ELIZA-style small talk rules ---------------------------------------

ELIZA_RULES = [
    (re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening))( there)?$"),
     "Hello! Tell me what is wrong and I will route you to the right team."),
    (re.compile(r"^(thanks|thank you|thx)( (so|very) much)?$"),
     "You're welcome. Is there anything else I can help with?"),
    (re.compile(r"^(bye|goodbye|see you)( later)?$"),
     "Goodbye! Have a nice day."),
    (re.compile(r"^how are you( doing| today)?$"),
     "I'm a support bot, always fine. How can I help you today?"),
    (re.compile(r"^are you (a )?(bot|robot|human|real)$"),
     "I'm an automated assistant. Ask for a human agent any time."),
]


def eliza_reply(normalized: str) -> str | None:
    """Whole-message rules only, so "thanks my card was declined" still gets routed."""
    for pattern, reply in ELIZA_RULES:
        if pattern.match(normalized):
            return reply
    return None
