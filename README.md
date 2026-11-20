# Customer Support Ticket Router (SWE015 Homework, Use Case 2)

A support-message router for an online shop. Each message is normalized, its intent is predicted with a confidence score, and the message is sent to one of 11 teams (orders, refunds, billing, account, delivery, …). Out-of-scope and low-confidence messages go to a **human agent**; greetings and thanks get a rule-based reply. Four models of increasing sophistication run side by side, and the **INT8-quantized BERT** makes the final decision.

![architecture](docs/architecture.png)

## Model ladder

| # | Model | Area | Precision |
|---|---|---|---|
| 1 | tf-idf (word 1–2 grams + character 2–5 grams) + logistic regression | B · classical NLP baseline | FP64 |
| 2 | Own skip-gram fastText (gensim, char 3–5 grams), averaged → feedforward net (256 hidden, ReLU, dropout), trained with SGD + backprop | A · neural network trained from scratch | FP32 |
| 3 | `bert-base-uncased` fine-tuned for 28-way classification (27 intents + out-of-scope) | C · Transformer | FP32 |
| 4 | Model 3 with dynamic INT8 quantization (`Linear` INT8, embeddings 8-bit weight-only) | D · efficient models | INT8 |

Ablation (also area D): `google/bert_uncased_L-4_H-256_A-4` (BERT-mini) distilled from model 3 (Hinton KD, T=2, α=0.5), before and after INT8.

Preprocessing is the same for all models (`src/common.py`, area B):
- regex normalization: URLs, e-mails, model codes (`DW-6534`) and order/invoice numbers become tokens; lower case; repeated letters collapsed
- minimum-edit-distance spelling correction against the training vocabulary
- ELIZA-style rules for greetings, and for thanks that do not ask for anything ("my order arrived, thanks!")

## Data

- **In-scope:** [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (CDLA-Sharing-1.0): 27 intents in 11 categories, used as the 11 teams. Duplicates removed; slots such as `{{Order Number}}` filled with realistic values. Stratified 70/10/20 split with a fixed seed.
- **Training negatives:** 3,000 requests from other domains of [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) (CC BY 3.0), e.g. "set a timer", labelled out-of-scope. Banking, credit-card and order intents are excluded because they overlap with our teams.
- **Shortcut fix:** Bitext sentences almost never name a product, while CLINC negatives often do, so the models first learned "a product name means out-of-scope". A product name is appended to 30% of the in-scope training sentences ("… for my blender"); validation and test are untouched.
- **Validation/test out-of-scope:** CLINC150's own out-of-scope queries ("what's the population of finland"), never used for training; 13% of each split.

Splits: train 20,244 (3,000 negatives) · validation 2,832 (369 out-of-scope) · test 5,659 (731 out-of-scope). A message goes to the human agent if the predicted class is out-of-scope or the top probability is below a per-model threshold τ chosen on validation. Every number below comes from `test`.

## Results

`python evaluate.py` regenerates [`results/results.md`](results/results.md). Misclassified test messages are listed in `results/errors.csv`.

Test split: Bitext 20% + CLINC150 out-of-scope, 5659 messages (731 out-of-scope). Latency: mean of 200 single-message calls incl. preprocessing, Apple M4 CPU, 4 torch threads, batch size 1.

| model                | precision      |   accuracy |   in-scope acc |   OOS recall |   size (MB) |   latency (ms) |   threshold |
|:---------------------|:---------------|-----------:|---------------:|-------------:|------------:|---------------:|------------:|
| tf-idf + LogReg      | FP64           |      99.58 |          99.76 |        98.36 |         8.2 |           0.73 |         0   |
| fastText + FFN       | FP32           |      99.58 |          99.76 |        98.36 |        83   |           0.36 |         0   |
| BERT-base fine-tuned | FP32           |      99.28 |          99.68 |        96.58 |       438   |          46.4  |         0.4 |
| BERT-mini distilled  | FP32           |      99.28 |          99.76 |        96.03 |        44.7 |           2.54 |         0.9 |
| BERT-mini distilled  | INT8 (dynamic) |      99.17 |          99.76 |        95.21 |        11.5 |           2.2  |         0.8 |
| BERT-base fine-tuned | INT8 (dynamic) |      99.24 |          99.57 |        96.99 |       110.3 |          13.9  |         0   |

Rows 4 and 5 are an ablation: the distilled BERT-mini before and after INT8.
- **The test set does not separate the models.** Bitext sentences are written from templates, so every model is above 99% on its own test split, and tf-idf is as good as BERT here.
- **Natural messages do separate them.** Typed the way a customer writes ("where is my order #48213? it was supposed to arrive monday", "I want my money back, the blender arrived broken"), BERT routes them correctly while the fastText network sends them to a human. Some messages still fail because no intent covers them ("I received the wrong size, I want to exchange it") or because the phrasing is far from Bitext ("my package never came").
- **Quantization:** BERT-base INT8 is 4× smaller and about 3× faster for −0.04 points, so it is the model in the app.
- **Distillation:** BERT-mini is 10× smaller and much faster again, but catches fewer out-of-scope messages (95.2% vs 97.0% after INT8), so it stays an ablation.
- Latency was measured on a laptop on battery with other processes running; it moves between runs (BERT-base FP32 measured 26–46 ms).

## Setup and run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py        # trains all models into models/ (about 20 minutes on an Apple M4)
python evaluate.py     # writes results/
python app.py          # http://127.0.0.1:7860
```

Model weights are not stored in this repository. `python train.py` rebuilds them with fixed seeds. Trained weights (586 MB): [swe015-ticket-router-weights.zip](https://github.com/rkfirat/swe015-ticket-router/releases/download/v1.0/swe015-ticket-router-weights.zip). Unzip them into the repository root (they unpack into `models/`) to skip training.

## Layout

```
app.py            Gradio application
train.py          trains tfidf / ffn / bert / distill stages
evaluate.py       same held-out test set for all models: accuracy, OOS recall, size, latency
src/common.py     data loading, normalization, edit-distance correction, ELIZA rules
src/models.py     the models behind one predict_proba interface, INT8 quantization
docs/             architecture diagram (Graphviz source + PNG), slide builder
results/          evaluation outputs and training log
```
