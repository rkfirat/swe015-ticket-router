Test split: Bitext 20% + CLINC150 out-of-scope, 5659 messages (731 out-of-scope). Latency: mean of 200 single-message calls incl. preprocessing, Apple M4 CPU, 4 torch threads, batch size 1.

| model                | precision      |   accuracy |   in-scope acc |   OOS recall |   size (MB) |   latency (ms) |   threshold |
|:---------------------|:---------------|-----------:|---------------:|-------------:|------------:|---------------:|------------:|
| tf-idf + LogReg      | FP64           |      99.58 |          99.76 |        98.36 |         8.2 |           0.73 |         0   |
| fastText + FFN       | FP32           |      99.58 |          99.76 |        98.36 |        83   |           0.36 |         0   |
| BERT-base fine-tuned | FP32           |      99.28 |          99.68 |        96.58 |       438   |          46.4  |         0.4 |
| BERT-mini distilled  | FP32           |      99.28 |          99.76 |        96.03 |        44.7 |           2.54 |         0.9 |
| BERT-mini distilled  | INT8 (dynamic) |      99.17 |          99.76 |        95.21 |        11.5 |           2.2  |         0.8 |
| BERT-base fine-tuned | INT8 (dynamic) |      99.24 |          99.57 |        96.99 |       110.3 |          13.9  |         0   |
