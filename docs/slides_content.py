"""Slide text for docs/make_slides.py. @@RESULTS@@ and @@HARDWARE@@ are filled from results/."""

NAME = "Recep Kamil F\\i{}rat"
import os

STUDENT_ID = os.environ.get("STUDENT_ID", "")  # kept out of the public repository
REPO = "https://github.com/rkfirat/swe015-ticket-router"
VIDEO = os.environ.get("VIDEO_URL", "VIDEO_LINK")

PREAMBLE = r"""\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{default}
\usecolortheme{seahorse}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{fontspec}
\usepackage{hyperref}
\hyphenpenalty=10000 \exhyphenpenalty=10000
"""

FRAMES = [
    ("Customer Support Ticket Router", rf"""
\centering\vspace{{4mm}}
{{\Large SWE015 -- Introduction to Large Language Models}}\\[2mm]
Homework: Building an Intelligent Application -- \textbf{{Use Case 2}}\\[8mm]
\begin{{tabular}}{{ll}}
Name & {NAME} \\
Student ID & {STUDENT_ID} \\
Repository & \url{{{REPO}}} \\
Demo video & \url{{{VIDEO}}} \\
\end{{tabular}}
"""),

    ("Introduction: the problem", r"""
\begin{itemize}
\item A support desk receives thousands of free-text \textbf{Turkish} messages a day; each must reach the right team \textbf{instantly}.
\item Users: the support team (routing) and customers (fast answer).
\item The app takes a message $\rightarrow$ normalizes it $\rightarrow$ predicts the intent with a confidence $\rightarrow$ routes it to one of 14 teams.
\item Low-confidence messages (e.g.\ topics never seen in training) go to a \textbf{human agent}; greetings and thanks get a rule-based reply.
\item Domain change (allowed by Section 4): Turkish \textbf{MASSIVE} assistant requests (alarm, smart home, transport, \dots) instead of appliance tickets -- public and labeled.
\end{itemize}
\textbf{Success} = high in-scope accuracy \emph{and} high out-of-scope recall (a wrong team is worse than a human agent), at CPU latency of a few ms.
"""),

    ("Development: data", r"""
\begin{itemize}
\item \textbf{MASSIVE 1.1, tr-TR} (FitzGerald et al., 2022, CC BY 4.0): 18 scenarios, 60 intents, fixed train/validation/test splits.
\item No out-of-scope class, so 4 scenarios (cooking, takeaway, social, weather) are \textbf{removed from training}; their validation/test sentences are labelled out-of-scope.
\item Result: 53 intents in 14 teams. Train 10,082 $\cdot$ validation 2,033 (281 OOS) $\cdot$ \textbf{test 2,974 (391 OOS)}.
\item Models trained on train; checkpoints and thresholds chosen on validation; \textbf{every number on the following slides is from the same test split}.
\item Preprocessing (identical for all models): NFKC, URL/e-mail/model-code tokens (\texttt{DW-6534}), Turkish lowercase (I$\rightarrow$ı, İ$\rightarrow$i), repeated letters, punctuation, restoring Turkish letters (\textit{isiklari} $\rightarrow$ \textit{ışıkları}), edit-distance spelling correction.
\end{itemize}
"""),

    ("Development: architecture", r"""
\centering
\vspace*{\fill}\includegraphics[width=\linewidth,height=0.85\textheight,keepaspectratio]{architecture.png}\vspace*{\fill}
"""),

    ("Development: technique map", r"""
\small
\begin{tabular}{>{\raggedright}p{2.4cm}>{\raggedright}p{6.2cm}>{\raggedright\arraybackslash}p{4.2cm}}
\toprule
\textbf{Area} & \textbf{Technique used} & \textbf{Where in the code} \\ \midrule
A $\cdot$ Neural networks & Feedforward net on averaged fastText vectors, trained with SGD (momentum) + backprop & \texttt{train.py: train\_ffn}, \texttt{src/models.py: FFN} \\
B $\cdot$ Classical NLP & Regex normalization, model-code extraction, min.\ edit distance spelling correction, ELIZA-style rules, tf-idf + logistic regression; own fastText embeddings & \texttt{src/common.py}, \texttt{train.py: train\_tfidf} \\
C $\cdot$ Transformers & BERTurk fine-tuned with a 53-way classification head; WordPiece tokenizer & \texttt{train.py: train\_bert} \\
D $\cdot$ Efficient models & Knowledge distillation BERTurk $\rightarrow$ Turkish BERT-mini, then dynamic INT8 quantization; size and latency benchmark & \texttt{train.py: train\_distill}, \texttt{src/models.py}, \texttt{evaluate.py} \\
\bottomrule
\end{tabular}
"""),

    ("Development: the four models and how they were trained", r"""
\small
\begin{enumerate}
\item \textbf{Classical baseline:} tf-idf on word 1--2 grams + character 2--5 grams (Turkish suffixes share stems) + logistic regression ($C{=}20$).
\item \textbf{Own neural network:} skip-gram fastText (200-d, char 3--5 grams, 40 epochs) trained on the training sentences; averaged sentence vector $\rightarrow$ FFN (256 ReLU, dropout 0.3) $\rightarrow$ softmax; SGD lr 0.1, momentum 0.9, 60 epochs, best validation epoch kept.
\item \textbf{Transformer:} BERTurk (\texttt{dbmdz/bert-base-turkish-uncased}, 110M params, \texttt{strip\_accents=False}) fine-tuned 4 epochs, AdamW lr $4\cdot10^{-5}$, batch 32, linear warm-up/decay (Apple M4 GPU).
\item \textbf{Quantized Transformer:} Turkish BERT-mini student (\texttt{ytu-ce-cosmos}, 4 layers, 256 hidden) trained 25 epochs on $0.5\cdot$KL(teacher, T=2) + $0.5\cdot$CE; then PyTorch dynamic quantization: \texttt{Linear} weights INT8, embedding table 8-bit weight-only.
\end{enumerate}
Routing: if $\max p < \tau$ $\rightarrow$ human agent (no model has an out-of-scope output). $\tau$ per model maximizes accuracy on validation (14\% out-of-scope, about the same share as test).
"""),

    ("Development: results (same held-out test set, 2,974 messages)", r"""
\centering\footnotesize
@@RESULTS@@

\vspace{3mm}
\raggedright\tiny @@HARDWARE@@ Size = files on disk needed by the model. Rows 4 and 5 are ablations (quantization of the teacher; student before quantization).
"""),

    ("Conclusion: what the comparison showed", r"""
\small
\begin{itemize}
\item \textbf{Out-of-scope is the hard part}: every model reaches 75--84\% in-scope accuracy, but catches only 38--55\% of never-seen topics. Unseen topics get confident wrong answers, mostly into the catch-all intent \texttt{general\_quirky}.
\item \textbf{The classical baseline is strong}: tf-idf + LogReg (74.9\%) beats my fastText + FFN (71.5\%) and has the best OOS recall (55\%). Character n-grams handle Turkish suffixes; averaging word vectors loses word order.
\item \textbf{The student beat its teacher}: BERTurk peaked at epoch 3 (79.1\% in-scope); the distilled BERT-mini reached 83.9\%. Soft labels regularize, and the student trained for 25 epochs.
\item \textbf{Distillation brought the speed}: 9.6$\times$ smaller and 15$\times$ faster than BERTurk.
\item \textbf{Quantization brought the size}: INT8 cuts the student from 46.3 to 11.9 MB (3.9$\times$) for $-2.6$ points. On the M4 CPU it is \emph{slower} for this small model (1.9 $\rightarrow$ 2.7 ms): quantizing activations at every layer costs more than the 256-wide multiplications save. On BERTurk, INT8 is 4$\times$ smaller and 2.1$\times$ faster for $-0.7$ points.
\item \textbf{Final app model}: 11.9 MB, 2.7 ms, 75.3\% accuracy -- 37$\times$ smaller and 11$\times$ faster than BERTurk, and slightly more accurate.
\end{itemize}

"""),

    ("Conclusion: failure examples and limitations", r"""
\small
Final model: 736 of 2,974 test messages wrong. \textbf{221} in-scope messages went to a human agent (the safe error), \textbf{221} out-of-scope messages went to a team, and 294 went to the wrong team.
\begin{itemize}
\item \textit{``bugün hava nasıl olacak''} (weather, never seen) $\rightarrow$ \texttt{general\_quirky} (0.88). The catch-all intent absorbs unknown topics with high confidence, so the threshold cannot catch it.
\item \textit{``bana makarna tarifi ver''} (cooking, never seen) $\rightarrow$ \texttt{general\_quirky} (0.64). This is the same failure.
\item \textit{``beni yarım saat sonra kaldır''} (true: \texttt{alarm\_set}) $\rightarrow$ human agent. ``kaldır'' means both ``wake'' and ``remove'' in Turkish.
\end{itemize}
\textbf{Limitations:}
\begin{itemize}
\item MASSIVE is assistant requests, not real support tickets, and intents stand in for teams.
\item Out-of-scope is simulated with 4 held-out scenarios.
\item Latency was measured on a laptop with other processes running, so the tf-idf and BERTurk numbers moved by up to 4$\times$ between runs.
\end{itemize}

"""),

    ("Conclusion: next steps", r"""
\begin{itemize}
\item Collect real Turkish support tickets (appliance faults, model codes) and map intents to real teams.
\item Train with an explicit out-of-scope class instead of relying on the threshold alone.
\item Try static INT8 or ONNX Runtime kernels, where INT8 actually speeds up the matrix multiplications on ARM.
\item Calibrate confidences (temperature scaling) so the human-agent threshold is more reliable.
\end{itemize}
"""),

    ("AI usage", r"""
\begin{itemize}
\item \textbf{Claude (Anthropic), via Claude Code}: wrote and debugged the training, evaluation and Gradio code, the Graphviz architecture diagram and the draft of these slides.
\item All training and evaluation ran on my own laptop (Apple M4); every number comes from these runs (\texttt{results/results.csv}, \texttt{results/train.log}), none from papers, model cards or AI answers.
\item No AI tool was used to generate data or labels; the dataset is public MASSIVE.
\end{itemize}
"""),
]
