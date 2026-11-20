# Customer Support Ticket Router (SWE015 Homework, Use Case 2)

A support-message router. Each message is normalized, its intent is predicted with a confidence score, and the message is sent to the right team. Low-confidence and out-of-scope messages go to a **human agent**, and greetings or thanks get a rule-based reply. Four models of increasing sophistication run side by side, and the **distilled, INT8-quantized Transformer** makes the final decision.

![architecture](docs/architecture.png)

## Model ladder

| # | Model | Area | Precision |
|---|---|---|---|
| 1 | tf-idf (word 1–2 grams) + logistic regression | B · classical NLP baseline | FP64 |
| 2 | Own skip-gram word2vec (gensim), averaged → feedforward net (256 hidden, ReLU, dropout), trained with SGD + backprop | A · neural network trained from scratch | FP32 |
| 3 | `bert-base-uncased` fine-tuned for 151-way intent classification | C · Transformer | FP32 |
| 4 | `google/bert_uncased_L-4_H-256_A-4` (BERT-mini) distilled from #3 (Hinton KD, T=2, α=0.5), then dynamic INT8 quantization of all `Linear` layers | D · efficient models | INT8 |

Preprocessing is the same for all four models (`src/common.py`, area B):
- regex normalization: URL/e-mail/model codes become tokens, lowercase, repeated letters collapsed
- model-code extraction (e.g. `DW-6534`)
- minimum-edit-distance spelling correction against the training vocabulary; real English words are not changed
- ELIZA-style whole-message rules for small talk

## Data

[CLINC150](https://huggingface.co/datasets/clinc/clinc_oos), `plus` configuration (Larson et al., 2019, CC BY 3.0). It has 150 intents plus an out-of-scope class. Splits:
- train: 15,250 messages
- validation: 3,100 messages
- test: 5,500 messages, of which 1,000 are out-of-scope

All models are trained on `train`. The confidence threshold for each model is tuned on `validation`. Every number in the results comes from `test`.

## Results

See [`results/results.md`](results/results.md); `python evaluate.py` regenerates it. Misclassified test messages are listed in `results/errors.csv`.

Test split: CLINC150 'plus' test, 5500 messages (1000 out-of-scope). Latency: mean of 200 single-message calls incl. preprocessing, Apple M4 CPU, 4 torch threads, batch size 1.

| model                | precision      |   accuracy |   in-scope acc |   OOS recall |   size (MB) |   latency (ms) |   threshold |
|:---------------------|:---------------|-----------:|---------------:|-------------:|------------:|---------------:|------------:|
| tf-idf + LogReg      | FP64           |      83.44 |          81.82 |         90.7 |        39   |           1.06 |        0.5  |
| word2vec + FFN       | FP32           |      81.49 |          80.33 |         86.7 |         4.8 |           0.89 |        0.8  |
| BERT-base fine-tuned | FP32           |      91.33 |          91.16 |         92.1 |       438.4 |          17.8  |        0.7  |
| BERT-base fine-tuned | INT8 (dynamic) |      91.27 |          91.09 |         92.1 |       110.4 |          15.05 |        0.65 |
| BERT-mini distilled  | FP32           |      89.51 |          90.18 |         86.5 |        44.8 |           1.85 |        0.75 |
| BERT-mini distilled  | INT8 (dynamic) |      88.96 |          89    |         88.8 |        11.6 |           2.83 |        0.8  |

Rows 4 and 5 are ablations: the quantized teacher, and the student before quantization.
- Distillation gives the speed: 9.6× faster than BERT-base, at 98% of its accuracy.
- INT8 quantization gives the size: 3.9× smaller, −0.55 points.
- On the M4 CPU, dynamic INT8 does not speed up the small student. The run-time activation quantization costs more than the 256-wide layers save.

## Setup and run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py        # trains all four models into models/ (about 20 minutes on an Apple M4)
python evaluate.py     # writes results/
python app.py          # http://127.0.0.1:7860
```

Model weights are not stored in this repository. `python train.py` rebuilds them with fixed seeds. Trained weights: WEIGHTS_LINK. Unzip them into `models/` to skip training.

## Layout

```
app.py            Gradio application
train.py          trains tfidf / ffn / bert / distill stages
evaluate.py       same held-out test set for all models: accuracy, OOS recall, size, latency
src/common.py     data loading, regex normalization, edit-distance spelling correction, ELIZA rules
src/models.py     the four models behind one predict_proba interface, INT8 quantization
docs/             architecture diagram (Graphviz source + PNG)
results/          evaluation outputs and training log
```
