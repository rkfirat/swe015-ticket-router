Test split: MASSIVE 1.1 tr-TR test, 2974 messages (391 out-of-scope from held-out scenarios). Latency: mean of 200 single-message calls incl. preprocessing, Apple M4 CPU, 4 torch threads, batch size 1.

| model                       | precision      |   accuracy |   in-scope acc |   OOS recall |   size (MB) |   latency (ms) |   threshold |
|:----------------------------|:---------------|-----------:|---------------:|-------------:|------------:|---------------:|------------:|
| tf-idf + LogReg             | FP64           |      74.88 |          77.86 |        55.24 |        25.5 |           5.28 |        0.45 |
| fastText + FFN              | FP32           |      71.52 |          74.72 |        50.38 |        86.9 |           1.04 |        0.4  |
| BERTurk fine-tuned          | FP32           |      74.41 |          79.13 |        43.22 |       442.7 |          29.18 |        0.5  |
| BERTurk fine-tuned          | INT8 (dynamic) |      73.74 |          78.28 |        43.73 |       111.4 |          13.58 |        0.5  |
| Turkish BERT-mini distilled | FP32           |      77.84 |          83.89 |        37.85 |        46.3 |           1.92 |        0.5  |
| Turkish BERT-mini distilled | INT8 (dynamic) |      75.25 |          80.06 |        43.48 |        11.9 |           2.73 |        0.45 |
