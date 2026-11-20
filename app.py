"""Müşteri destek yönlendiricisi - Gradio uygulaması.

    python app.py      # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import time

import gradio as gr
import pandas as pd

from src.common import RESULTS, TO_HUMAN, eliza_reply, extract_model_codes, load_data, normalize
from src.models import load_all

_, NAMES, OOS, TEAM_OF = load_data()
MODELS = load_all()
ROUTER = MODELS[-1]  # the quantized Transformer makes the routing decision (R5)
THRESHOLDS = json.loads((RESULTS / "thresholds.json").read_text()) if (RESULTS / "thresholds.json").exists() else {}

EXAMPLES = [
    "yarın sabah yedide beni uyandır",
    "isiklari kapatirmisin",
    "ankaraya yarın için tren bileti bul",
    "bulaşık makinem DW-6534 su boşaltmıyor",
    "merhaba",
    "yarın istanbulda yağmur yağacak mı",
    "bana bir fıkra anlat",
]


def classify(text: str):
    text = (text or "").strip()
    if not text:
        return "", "", "Bir mesaj yaz.", pd.DataFrame()

    norm = normalize(text)
    codes = ", ".join(extract_model_codes(text)) or "-"

    rows, decision = [], None
    for m in MODELS:
        key = f"{m.name} [{m.precision}]"
        tau = THRESHOLDS.get(key, 0.5)
        t0 = time.perf_counter()
        probs = m.predict_proba([text])[0]
        ms = (time.perf_counter() - t0) * 1000
        idx = int(probs.argmax())
        conf = float(probs[idx])
        if conf < tau:
            routed, reason = "müşteri temsilcisi", f"düşük güven {conf:.2f} < eşik {tau:.2f}"
        elif NAMES[idx] in TO_HUMAN:
            routed, reason = "müşteri temsilcisi", f"belirsiz istek ({NAMES[idx]}), sahibi olan ekip yok"
        else:
            routed, reason = TEAM_OF[NAMES[idx]], f"niyet {NAMES[idx]}, güven {conf:.2f}"
        rows.append({"model": key, "tahmin edilen niyet": NAMES[idx], "güven": round(conf, 3),
                     "yönlendirme": routed, "gecikme (ms)": round(ms, 1)})
        if m is ROUTER:
            decision = (routed, reason)

    reply = eliza_reply(norm)
    if reply:
        answer = f"**Kural tabanlı cevap (sohbet):** {reply}"
    else:
        answer = f"**Yönlendirildi: {decision[0]}** ({decision[1]})"
    return norm, codes, answer, pd.DataFrame(rows)


with gr.Blocks(title="Destek Yönlendirici") as demo:
    gr.Markdown("# Müşteri Destek Yönlendiricisi\nMASSIVE tr-TR: 53 niyet, 14 ekip. Düşük güvenli mesajlar "
                "müşteri temsilcisine gider. Son kararı damıtılmış INT8 Türkçe BERT-mini verir; dört model yan yana.")
    with gr.Row():
        inp = gr.Textbox(label="Müşteri mesajı", lines=2, scale=4)
        btn = gr.Button("Yönlendir", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inp, label="Örnekler")
    decision = gr.Markdown()
    with gr.Row():
        norm = gr.Textbox(label="Normalize + yazım düzeltilmiş girdi")
        codes = gr.Textbox(label="Yakalanan model kodları")
    table = gr.Dataframe(label="Tüm modeller", interactive=False)
    btn.click(classify, inp, [norm, codes, decision, table])
    inp.submit(classify, inp, [norm, codes, decision, table])

if __name__ == "__main__":
    demo.launch()
