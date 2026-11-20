"""Customer Support Ticket Router - Gradio app.

    python app.py      # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import time

import gradio as gr
import pandas as pd

from src.common import RESULTS, eliza_reply, extract_model_codes, load_clinc, normalize
from src.models import load_all

_, NAMES, OOS = load_clinc()
MODELS = load_all()
ROUTER = MODELS[-1]  # the quantized Transformer makes the routing decision (R5)
THRESHOLDS = json.loads((RESULTS / "thresholds.json").read_text()) if (RESULTS / "thresholds.json").exists() else {}

EXAMPLES = [
    "my card got declined at the grocery store",
    "whats my acount balanse",
    "i neeed to book a fligt to boston tomorow",
    "the dishwasher DW-6534 is not draining, what do I do",
    "who won the football game last night",
    "hello",
    "what is the official language of romania",
]


def pretty(intent: str) -> str:
    return intent.replace("_", " ")


def classify(text: str):
    text = (text or "").strip()
    if not text:
        return "", "", "Type a message.", pd.DataFrame()

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
        if idx == OOS:
            routed, reason = "human agent", f"predicted out-of-scope, p={conf:.2f}"
        elif conf < tau:
            routed, reason = "human agent", f"low confidence {conf:.2f} < {tau:.2f}"
        else:
            routed, reason = pretty(NAMES[idx]), f"confidence {conf:.2f}"
        rows.append({"model": key, "predicted intent": pretty(NAMES[idx]),
                     "confidence": round(conf, 3), "routed to": routed, "latency (ms)": round(ms, 1)})
        if m is ROUTER:
            decision = (routed, reason)

    reply = eliza_reply(norm)
    if reply:
        answer = f"**Rule-based reply (small talk):** {reply}"
    elif decision[0] == "human agent":
        answer = f"**Routed to: human agent** ({decision[1]})"
    else:
        answer = f"**Routed to team: {decision[0]}** ({decision[1]})"
    return norm, codes, answer, pd.DataFrame(rows)


with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Customer Support Ticket Router\nCLINC150 intents (150 + out-of-scope). "
                "The final decision comes from the distilled INT8 BERT-mini; all four models are shown side by side.")
    with gr.Row():
        inp = gr.Textbox(label="Customer message", lines=2, scale=4)
        btn = gr.Button("Route", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inp)
    decision = gr.Markdown()
    with gr.Row():
        norm = gr.Textbox(label="Normalized + spell-corrected input")
        codes = gr.Textbox(label="Extracted model codes")
    table = gr.Dataframe(label="All models", interactive=False)
    btn.click(classify, inp, [norm, codes, decision, table])
    inp.submit(classify, inp, [norm, codes, decision, table])

if __name__ == "__main__":
    demo.launch()
