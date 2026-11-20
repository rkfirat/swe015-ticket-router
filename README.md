# Customer Support Ticket Router (SWE015 Homework, Use Case 2)

A support-message router for **Turkish** messages. Each message is normalized, its intent is predicted with a confidence score, and the message is sent to one of 14 teams. Low-confidence messages go to a **human agent**, and greetings or thanks get a rule-based reply. Four models of increasing sophistication run side by side, and the **distilled, INT8-quantized Turkish Transformer** makes the final decision.

![architecture](docs/architecture.png)

## Model ladder

| # | Model | Area | Precision |
|---|---|---|---|
| 1 | tf-idf (word 1–2 grams + character 2–5 grams) + logistic regression | B · classical NLP baseline | FP64 |
| 2 | Own skip-gram fastText (gensim, char 3–5 grams), averaged → feedforward net (256 hidden, ReLU, dropout), trained with SGD + backprop | A · neural network trained from scratch | FP32 |
| 3 | BERTurk (`dbmdz/bert-base-turkish-uncased`, `strip_accents=False`) fine-tuned for 53-way intent classification | C · Transformer | FP32 |
| 4 | Turkish BERT-mini (`ytu-ce-cosmos/turkish-mini-bert-uncased`) distilled from #3 (Hinton KD, T=2, α=0.5), then dynamic INT8 quantization (`Linear` INT8, embeddings 8-bit weight-only) | D · efficient models | INT8 |

Preprocessing is the same for all four models (`src/common.py`, area B):
- regex normalization: URL/e-mail/model codes become tokens, Turkish lowercase (I→ı, İ→i), repeated letters collapsed
- model-code extraction (e.g. `DW-6534`)
- restoring missing Turkish letters (`isiklari` → `ışıkları`)
- minimum-edit-distance spelling correction against the training vocabulary
- ELIZA-style whole-message rules for small talk

## Data

[MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive), `tr-TR` locale (FitzGerald et al., 2022, CC BY 4.0). MASSIVE has no out-of-scope class, so four scenarios (`cooking`, `takeaway`, `social`, `weather`) are removed from training. Their validation and test sentences are labelled out-of-scope: topics the router has never seen. That leaves 53 intents in 14 teams. Splits:
- train: 10,082 sentences
- validation: 2,033 sentences, of which 281 are out-of-scope
- test: 2,974 sentences, of which 391 are out-of-scope

No model has an out-of-scope output: a message goes to the human agent when the top probability is below a per-model threshold τ. Checkpoints and τ are chosen on `validation`. Every number in the results comes from `test`.

## Results

`python evaluate.py` regenerates [`results/results.md`](results/results.md). Misclassified test messages are listed in `results/errors.csv`.

Test split: MASSIVE 1.1 tr-TR test, 2974 messages (391 out-of-scope from held-out scenarios). Latency: mean of 200 single-message calls incl. preprocessing, Apple M4 CPU, 4 torch threads, batch size 1.

| model                       | precision      |   accuracy |   in-scope acc |   OOS recall |   size (MB) |   latency (ms) |   threshold |
|:----------------------------|:---------------|-----------:|---------------:|-------------:|------------:|---------------:|------------:|
| tf-idf + LogReg             | FP64           |      74.88 |          77.86 |        55.24 |        25.5 |           5.28 |        0.45 |
| fastText + FFN              | FP32           |      71.52 |          74.72 |        50.38 |        86.9 |           1.04 |        0.4  |
| BERTurk fine-tuned          | FP32           |      74.41 |          79.13 |        43.22 |       442.7 |          29.18 |        0.5  |
| BERTurk fine-tuned          | INT8 (dynamic) |      73.74 |          78.28 |        43.73 |       111.4 |          13.58 |        0.5  |
| Turkish BERT-mini distilled | FP32           |      77.84 |          83.89 |        37.85 |        46.3 |           1.92 |        0.5  |
| Turkish BERT-mini distilled | INT8 (dynamic) |      75.25 |          80.06 |        43.48 |        11.9 |           2.73 |        0.45 |

Rows 4 and 5 are ablations: the quantized teacher, and the student before quantization.
- Out-of-scope detection is the weak point. Unseen topics often land in the catch-all intent `general_quirky` with high confidence.
- Distillation gives the speed: 15× faster and 9.6× smaller than BERTurk. Here the student is also more accurate, because the teacher peaked at epoch 3.
- INT8 quantization gives the size: 3.9× smaller, −2.6 points. On the M4 CPU it does not speed up the small student. The run-time activation quantization costs more than the 256-wide layers save.
- Latency was measured with other processes running on the laptop. Between runs, the tf-idf and BERTurk latencies moved by up to 4×.

## Setup and run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py        # trains all four models into models/ (about 20 minutes on an Apple M4)
python evaluate.py     # writes results/
python app.py          # http://127.0.0.1:7860
```

Model weights are not stored in this repository. `python train.py` rebuilds them with fixed seeds. Trained weights: WEIGHTS_LINK. Unzip them into the repository root (they unpack into `models/`) to skip training.

## Layout

```
app.py            Gradio application
train.py          trains tfidf / ffn / bert / distill stages
evaluate.py       same held-out test set for all models: accuracy, OOS recall, size, latency
src/common.py     data loading, normalization, Turkish letter restoration, edit-distance correction, ELIZA rules
src/models.py     the four models behind one predict_proba interface, INT8 quantization
docs/             architecture diagram (Graphviz source + PNG), slide builder
results/          evaluation outputs and training log
```
