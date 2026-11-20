"""Shared paths, data loading and the text normalizer (area B)."""
from __future__ import annotations

import json
import random
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

# No intent is sent to a human on top of the threshold rule; Bitext has its own
# contact_human_agent intent, which the "Human agent" team owns.
TO_HUMAN: tuple[str, ...] = ()

# Each Bitext category is one support team.
TEAMS = {
    "ACCOUNT": "Account team", "CANCEL": "Cancellation team", "CONTACT": "Human agent",
    "DELIVERY": "Delivery team", "FEEDBACK": "Feedback team", "INVOICE": "Billing team",
    "ORDER": "Orders team", "PAYMENT": "Payments team", "REFUND": "Refunds team",
    "SHIPPING": "Shipping address team", "SUBSCRIPTION": "Newsletter team",
}

# Bitext writes slots as {{Order Number}}; real customers type values, so we fill them.
SLOTS = {
    "{{Order Number}}": lambda r: r.choice(["#", "order ", "#", ""]) + str(r.randint(10000, 999999)),
    "{{Invoice Number}}": lambda r: f"INV-{r.randint(1000, 99999)}",
    "{{Account Type}}": lambda r: r.choice(["free", "premium", "pro", "business", "gold", "standard"]),
    "{{Account Category}}": lambda r: r.choice(["personal", "business", "family", "student", "platinum"]),
    "{{Person Name}}": lambda r: r.choice(["John", "Maria", "Ahmet", "Emma", "Lucas", "Ayse", "Olivia", "Noah"]),
    "{{Refund Amount}}": lambda r: str(r.choice([15, 29, 49, 120, 250, 999])),
    "{{Currency Symbol}}": lambda r: r.choice(["$", "€", "£"]),
    "{{Delivery City}}": lambda r: r.choice(["Istanbul", "London", "Berlin", "Madrid", "Chicago", "Lisbon"]),
    "{{Delivery Country}}": lambda r: r.choice(["Turkey", "Germany", "Spain", "the UK", "Portugal", "the US"]),
}
# Bitext sentences almost never name a product, while the CLINC negatives often do ("blender",
# "jacket"). Without help the models learn "a product name means out-of-scope". We append a
# product to 30% of the in-scope training sentences to break that shortcut (val/test untouched).
PRODUCTS = ["blender", "jacket", "phone", "laptop", "shoes", "headphones", "coffee machine", "sofa",
            "t-shirt", "watch", "camera", "vacuum cleaner", "lamp", "backpack", "jeans", "tv",
            "microwave", "toaster", "book", "keyboard", "package", "parcel", "dress", "mattress"]
PRODUCT_TAILS = [" (the {p})", " for my {p}", ", it's about the {p} i bought", " - the {p}"]


def augment_product(text: str, r: random.Random, share: float = 0.3) -> str:
    if r.random() >= share:
        return text
    return text + r.choice(PRODUCT_TAILS).format(p=r.choice(PRODUCTS))


# CLINC150 intents never used as negatives: they overlap with an online-shop support desk.
OVERLAPPING = {
    "transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance", "bill_due",
    "interest_rate", "routing", "min_payment", "order_checks", "pin_change", "report_fraud",
    "account_blocked", "spending_history", "credit_score", "report_lost_card", "credit_limit",
    "rewards_balance", "new_card", "application_status", "card_declined", "international_fees", "apr",
    "redeem_rewards", "credit_limit_change", "damaged_card", "replacement_card_duration",
    "improve_credit_score", "expiration_date", "order", "order_status", "cancel", "direct_deposit",
    # Short replies and complaint-like phrasing ("no, that is wrong", "thanks") that real support
    # messages also contain; as negatives they taught the models "natural phrasing = out-of-scope".
    "no", "yes", "maybe", "thank_you", "repeat", "greeting", "goodbye",
    "lost_luggage", "cancel_reservation", "confirm_reservation", "last_maintenance", "insurance",
}
SLOT = re.compile(r"\{\{[^}]+\}\}")


def fill_slots(text: str, r: random.Random) -> str:
    return SLOT.sub(lambda m: SLOTS.get(m.group(0), lambda _: "")(r), text)


@lru_cache(maxsize=1)
def load_data():
    """Bitext customer-support dataset (27 intents in 11 teams, CDLA-Sharing-1.0), split 70/10/20
    with a fixed seed. CLINC150 (CC BY 3.0) adds 3,000 other-domain training negatives and its
    out-of-scope queries for validation and test.

    Returns (splits, names, oos, team_of) where splits[k] = (texts, labels),
    names[i] is the intent name, names[oos] == "oos" and team_of[intent] is the team.
    """
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from sklearn.model_selection import train_test_split

    path = hf_hub_download("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                           "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
                           repo_type="dataset")
    df = pd.read_csv(path).drop_duplicates("instruction")
    r = random.Random(SEED)
    df["text"] = [fill_slots(t, r) for t in df["instruction"]]
    in_scope = sorted(df["intent"].unique())
    names = in_scope + ["oos"]
    oos = len(in_scope)
    index = {n: i for i, n in enumerate(in_scope)}
    team_of = dict(zip(df["intent"], (TEAMS[c] for c in df["category"])))
    y = [index[i] for i in df["intent"]]
    xtr, xrest, ytr, yrest = train_test_split(list(df["text"]), y, test_size=0.3, stratify=y, random_state=SEED)
    xva, xte, yva, yte = train_test_split(xrest, yrest, test_size=2 / 3, stratify=yrest, random_state=SEED)

    clinc = load_dataset("clinc/clinc_oos", "plus")
    cl_names = clinc["train"].features["intent"].names
    cl_oos = cl_names.index("oos")

    # Training negatives: CLINC150 requests from other domains ("set a timer", "how do i
    # cook rice"), labelled "oos" so the models learn what is NOT a support request.
    # Banking/credit-card and order intents are left out: they overlap with our teams.
    neg = [t for t, i in zip(clinc["train"]["text"], clinc["train"]["intent"])
           if i != cl_oos and cl_names[i] not in OVERLAPPING]
    r.shuffle(neg)
    neg = neg[:3000]
    xtr = [augment_product(t, r) for t in xtr]  # in-scope training sentences only
    xtr, ytr = xtr + neg, ytr + [oos] * len(neg)

    # Validation/test out-of-scope: CLINC150's own out-of-scope queries, which belong to
    # none of its 150 intents ("what's the population of finland"). Never trained on.
    pool = [t for k in ("validation", "test") for t, i in zip(clinc[k]["text"], clinc[k]["intent"]) if i == cl_oos]
    r.shuffle(pool)
    n_val = round(len(xva) * 0.15)
    xva, yva = xva + pool[:n_val], yva + [oos] * n_val
    xte, yte = xte + pool[n_val:], yte + [oos] * len(pool[n_val:])
    return {"train": (xtr, ytr), "validation": (xva, yva), "test": (xte, yte)}, names, oos, team_of


# --- B: regex normalization ------------------------------------------------

# Upper-case product/model codes like "DW-6534" or "RF5000".
MODEL_CODE = re.compile(r"\b([A-Z]{1,4})-?(\d{3,5}[A-Z]?)\b")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
NUMBER = re.compile(r"#?\b(?:inv-)?\d{3,}\b")  # order / invoice numbers -> <num>
REPEAT = re.compile(r"(\w)\1{2,}")  # "pleeease" -> "pleease"
NON_WORD = re.compile(r"[^a-z0-9<> ']+")
SPACES = re.compile(r"\s+")


def extract_model_codes(text: str) -> list[str]:
    return [f"{a}-{b}" for a, b in MODEL_CODE.findall(text)]


def regex_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = URL.sub(" <url> ", text)
    text = EMAIL.sub(" <email> ", text)
    text = MODEL_CODE.sub(" <model> ", text)
    text = text.lower()
    text = NUMBER.sub(" <num> ", text)
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
    """Real English words outside the training vocabulary ("blender") are left alone."""
    from wordfreq import zipf_frequency

    return zipf_frequency(w, "en") >= 3.0


class SpellCorrector:
    """Replaces unknown words with the closest frequent training word (minimum edit distance)."""

    def __init__(self, vocab: Counter, min_count: int = 2):
        self.freq = {w: c for w, c in vocab.items() if c >= min_count}
        self.by_len: dict[int, list[str]] = {}
        for w in sorted(self.freq, key=self.freq.get):  # most frequent wins
            self.by_len.setdefault(len(w), []).append(w)

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
        if len(w) < 4 or not w.isalpha() or is_english_word(w):
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
    (re.compile(r"^(hi+|hel+o+|hey+|good (morning|afternoon|evening))( there)?$"),
     "Hello! Tell me what you need and I will route you to the right team."),
    (re.compile(r"^(thanks|thank you|thx|ty)( (so|very) much)?( for (your|the) help)?$"),
     "You're welcome! Is there anything else I can help with?"),
    (re.compile(r"^(bye|goodbye|see you)( later)?$"), "Goodbye, have a nice day!"),
    (re.compile(r"^how are you( doing| today)?$"), "I'm a support bot, always fine. How can I help?"),
    (re.compile(r"^are you (a )?(bot|robot|human|real)$"), "I'm an automated assistant. Ask for a human agent any time."),
]
# "my order arrived, thanks!" is gratitude, not a ticket, unless it also asks for something.
THANKS = re.compile(r"\b(thanks|thank you|thx|appreciate it)\b")
REQUEST = re.compile(r"\b(not|never|no|wrong|broken|damaged|late|missing|cancel|refund|return|change|help|how|what|where|when|why|can|could|need|want|problem|issue)\b")


def eliza_reply(normalized: str) -> str | None:
    """Whole-message rules, plus thanks without any request in the same message."""
    for pattern, reply in ELIZA_RULES:
        if pattern.match(normalized):
            return reply
    if THANKS.search(normalized) and not REQUEST.search(normalized):
        return "Glad to hear that, thank you! Is there anything else I can help with?"
    return None
