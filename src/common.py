"""Shared paths, data loading and the Turkish text normalizer (area B)."""
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

# MASSIVE has no out-of-scope class. These scenarios are removed from training and their
# validation/test messages are labelled "oos": topics the router has never seen.
HELD_OUT = ("cooking", "takeaway", "social", "weather")

# MASSIVE's catch-all intent for odd requests ("bana bugünü anlat", "merkezi olmayan tasarım").
# No team owns it, and unseen topics land in it with high confidence, so it goes to a human.
TO_HUMAN = ("general_quirky",)

# Each MASSIVE scenario is one support team.
TEAMS = {
    "alarm": "Alarm ekibi", "audio": "Ses ekibi", "calendar": "Takvim ekibi",
    "datetime": "Tarih/saat ekibi", "email": "E-posta ekibi", "general": "Genel destek",
    "iot": "Akıllı ev ekibi", "lists": "Liste ekibi", "music": "Müzik ekibi",
    "news": "Haber ekibi", "play": "Medya oynatma ekibi", "qa": "Bilgi ekibi",
    "recommendation": "Öneri ekibi", "transport": "Ulaşım ekibi",
}


@lru_cache(maxsize=1)
def load_data():
    """MASSIVE 1.1 tr-TR (Amazon, CC BY 4.0) with held-out scenarios as out-of-scope.

    Returns (splits, names, oos, team_of) where splits[k] = (texts, labels),
    names[i] is the intent name, names[oos] == "oos" and team_of[intent] is the team.
    """
    from datasets import load_dataset

    base = "hf://datasets/AmazonScience/massive@refs%2Fconvert%2Fparquet/tr-TR/"
    ds = load_dataset("parquet", data_files={s: f"{base}{s}/0000.parquet"
                                             for s in ("train", "validation", "test")})
    scen = ds["train"].features["scenario"].names
    intents = ds["train"].features["intent"].names

    in_scope = sorted({intents[i] for i, s in zip(ds["train"]["intent"], ds["train"]["scenario"])
                       if scen[s] not in HELD_OUT})
    names = in_scope + ["oos"]
    oos = len(in_scope)
    index = {n: i for i, n in enumerate(in_scope)}
    team_of = {n: TEAMS[n.split("_")[0]] for n in in_scope}

    splits = {}
    for k in ("train", "validation", "test"):
        texts, labels = [], []
        for t, i, s in zip(ds[k]["utt"], ds[k]["intent"], ds[k]["scenario"]):
            if scen[s] in HELD_OUT:
                if k == "train":
                    continue  # never shown to any model
                labels.append(oos)
            else:
                labels.append(index[intents[i]])
            texts.append(t)
        splits[k] = (texts, labels)
    return splits, names, oos, team_of


# --- B: regex normalization ------------------------------------------------

# Upper-case product/model codes like "DW-6534" or "RF5000".
MODEL_CODE = re.compile(r"\b([A-Z]{1,4})-?(\d{3,5}[A-Z]?)\b")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
REPEAT = re.compile(r"(\w)\1{2,}")  # "çooook" -> "çook"
NON_WORD = re.compile(r"[^a-z0-9çğıöşü<> ]+")
SPACES = re.compile(r"\s+")


def tr_lower(text: str) -> str:
    """Turkish lower case: I -> ı and İ -> i (Python's lower() turns İ into i + a combining dot)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def extract_model_codes(text: str) -> list[str]:
    return [f"{a}-{b}" for a, b in MODEL_CODE.findall(text)]


def regex_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = URL.sub(" <url> ", text)
    text = EMAIL.sub(" <email> ", text)
    text = MODEL_CODE.sub(" <model> ", text)
    text = tr_lower(text)
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


def is_turkish_word(w: str) -> bool:
    """Real Turkish words outside the training vocabulary are left alone."""
    from wordfreq import zipf_frequency

    return zipf_frequency(w, "tr") >= 2.5


ASCII_FOLD = str.maketrans("çğıöşü", "cgiosu")


class SpellCorrector:
    """Fixes unknown words: first missing Turkish letters ("muzik" -> "müzik"),
    then the closest frequent training word by minimum edit distance."""

    def __init__(self, vocab: Counter, min_count: int = 2):
        self.freq = {w: c for w, c in vocab.items() if c >= min_count}
        self.by_len: dict[int, list[str]] = {}
        self.folded: dict[str, str] = {}
        for w in sorted(self.freq, key=self.freq.get):  # most frequent wins
            self.by_len.setdefault(len(w), []).append(w)
            self.folded[w.translate(ASCII_FOLD)] = w

    @classmethod
    def from_texts(cls, texts) -> "SpellCorrector":
        return cls(Counter(w for t in texts for w in regex_normalize(t).split()))

    def save(self, path: Path):
        path.write_text(json.dumps(self.freq, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "SpellCorrector":
        return cls(Counter(json.loads(path.read_text())), min_count=0)

    def correct_word(self, w: str) -> str:
        if w in self.freq:
            return w
        if w in self.folded:
            return self.folded[w]
        if len(w) < 4 or not w.isalpha() or is_turkish_word(w):
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
    splits, _, _, _ = load_data()
    sc = SpellCorrector.from_texts(splits["train"][0])
    sc.save(path)
    return sc


def normalize(text: str, correct: bool = True) -> str:
    text = regex_normalize(text)
    return get_corrector()(text) if correct else text


# --- B: ELIZA-style small talk rules ---------------------------------------

ELIZA_RULES = [
    (re.compile(r"^(merhaba|selam|selamlar|günaydın|iyi akşamlar|iyi günler)$"),
     "Merhaba! Sorununu yaz, seni doğru ekibe yönlendireyim."),
    (re.compile(r"^(teşekkürler|teşekkür ederim|sağ ol|sağol|eyvallah)( çok)?$"),
     "Rica ederim. Başka bir konuda yardımcı olabilir miyim?"),
    (re.compile(r"^(görüşürüz|hoşça kal|güle güle|bay bay)$"),
     "Görüşmek üzere, iyi günler!"),
    (re.compile(r"^(nasılsın|naber|ne haber)$"),
     "Ben bir destek botuyum, hep iyiyim. Sana nasıl yardımcı olabilirim?"),
    (re.compile(r"^(sen )?(bot|robot|insan) mısın$"),
     "Otomatik bir asistanım. İstersen seni bir müşteri temsilcisine aktarırım."),
]


def eliza_reply(normalized: str) -> str | None:
    """Whole-message rules only, so "teşekkürler kartım reddedildi" still gets routed."""
    for pattern, reply in ELIZA_RULES:
        if pattern.match(normalized):
            return reply
    return None
