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
\item A support desk receives thousands of free-text messages a day (orders, refunds, invoices, accounts); each must reach the right team \textbf{instantly}.
\item Users: the support team (routing) and customers (fast answer).
\item The app takes a message $\rightarrow$ normalizes it $\rightarrow$ predicts the intent with a confidence $\rightarrow$ routes it to one of 11 teams.
\item Low-confidence or out-of-scope messages go to a \textbf{human agent}; greetings and thanks get a rule-based reply.
\item Domain (allowed by Section 4): a general online-shop support desk in English instead of an appliance brand.
\end{itemize}
\textbf{Success} = high in-scope accuracy \emph{and} high out-of-scope recall (a wrong team is worse than a human agent), at CPU latency of a few ms.
"""),

    ("Development: data", r"""
\begin{itemize}
\item \textbf{Bitext customer-support dataset} (CDLA-Sharing-1.0): 27 intents in 11 categories, used as the 11 teams (orders, refunds, invoices, account, delivery, \dots). Duplicates removed; slots such as \texttt{\{\{Order Number\}\}} filled with realistic values.
\item Stratified split with a fixed seed: train 70\% $\cdot$ validation 10\% $\cdot$ test 20\%.
\item \textbf{Out-of-scope:} 3,000 other-domain CLINC150 requests (Larson et al., 2019, CC BY 3.0) as a trained out-of-scope class (``set a timer''); CLINC150's own out-of-scope queries (``what's the population of finland'') only in validation and test, 13\% of each.
\item Result: train 20,244 (3,000 negatives) $\cdot$ validation 2,832 (369 OOS) $\cdot$ \textbf{test 5,659 (731 OOS)}. Every number on the following slides is from the same test split.
\item Preprocessing (identical for all models): NFKC, URL/e-mail/model-code/number tokens, lower case, repeated letters, punctuation, edit-distance spelling correction.
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
C $\cdot$ Transformers & BERT-base fine-tuned with a 27-way classification head; WordPiece tokenizer & \texttt{train.py: train\_bert} \\
D $\cdot$ Efficient models & Dynamic INT8 quantization of BERT-base (app model); distillation into BERT-mini as ablation; size and latency benchmark & \texttt{train.py: train\_distill}, \texttt{src/models.py}, \texttt{evaluate.py} \\
\bottomrule
\end{tabular}
"""),

    ("Development: the four models and how they were trained", r"""
\small
\begin{enumerate}
\item \textbf{Classical baseline:} tf-idf on word 1--2 grams + character 2--5 grams (typos still share pieces) + logistic regression ($C{=}20$).
\item \textbf{Own neural network:} skip-gram fastText (200-d, char 3--5 grams, 40 epochs) trained on the training sentences; averaged sentence vector $\rightarrow$ FFN (256 ReLU, dropout 0.3) $\rightarrow$ softmax; SGD lr 0.1, momentum 0.9, 60 epochs, best validation epoch kept.
\item \textbf{Transformer:} \texttt{bert-base-uncased} (110M params) fine-tuned 3 epochs for 27 intents + out-of-scope, AdamW lr $4\cdot10^{-5}$, batch 32, linear warm-up/decay (Apple M4 GPU).
\item \textbf{Quantized Transformer:} the same BERT-base with PyTorch dynamic quantization: \texttt{Linear} weights INT8, embedding table 8-bit weight-only.
\end{enumerate}
Ablation: BERT-mini student (\texttt{google/bert\_uncased\_L-4\_H-256\_A-4}) distilled from BERT-base, 15 epochs on $0.5\cdot$KL(teacher, T=2) + $0.5\cdot$CE, then INT8.

Routing: predicted out-of-scope or $\max p < \tau$ $\rightarrow$ human agent; $\tau$ per model chosen on validation.
"""),

    ("Development: results (same held-out test set, 5,659 messages)", r"""
\centering\footnotesize
@@RESULTS@@

\vspace{3mm}
\raggedright\tiny @@HARDWARE@@ Size = files on disk needed by the model. Rows 4 and 5 are ablations (quantization of the teacher; student before quantization).
"""),

    ("Conclusion: what the comparison showed", r"""
\small
\begin{itemize}
\item \textbf{The test set does not separate the models.} Bitext sentences come from templates, so every model is above 99\%; tf-idf (99.6\%) is as good as BERT (99.3\%).
\item \textbf{Natural messages do.} ``where is my order \#48213? it was supposed to arrive monday'' and ``I want my money back, the blender arrived broken'' reach the right team with BERT; the fastText network sends both to a human.
\item \textbf{The out-of-scope class matters most.} With only a threshold, BERT gave ``what is the capital of france'' 0.89 confidence for a delivery intent. 3,000 other-domain negatives fixed it; I had to drop CLINC intents such as ``no'' and ``thank\_you'' and add product names to Bitext sentences, because the models first learned ``natural phrasing = out-of-scope''.
\item \textbf{Quantization:} BERT-base INT8 is 4$\times$ smaller and about 3$\times$ faster for $-0.04$ points, so it runs in the app.
\item \textbf{Distillation:} BERT-mini is 10$\times$ smaller and much faster, but catches fewer out-of-scope messages (95.2\% vs 97.0\%), so it stays an ablation.
\end{itemize}
"""),

    ("Conclusion: failure examples and limitations", r"""
\small
\begin{itemize}
\item \textit{``I received the wrong size, I want to exchange it''} $\rightarrow$ human agent. None of the 27 intents covers exchanges, so this is the right fallback.
\item \textit{``my package never came''} $\rightarrow$ human agent (out-of-scope 0.82). The phrasing is far from Bitext's ``how long does delivery take''.
\item \textit{``I was charged twice for the same purchase''} $\rightarrow$ human agent instead of the payments team.
\end{itemize}
\textbf{Limitations:}
\begin{itemize}
\item Bitext is template-based, so test accuracy overstates real-world accuracy.
\item Out-of-scope test queries come from another dataset (CLINC150).
\item Latency was measured on a laptop on battery with other processes running; BERT-base FP32 measured 26--46 ms in different runs.
\end{itemize}
"""),

    ("Conclusion: next steps", r"""
\begin{itemize}
\item Collect real support tickets and label a test set in the customers' own words.
\item Add intents the data lacks (exchanges, damaged items) and a Turkish version for the original scenario.
\item Try static INT8 or ONNX Runtime kernels, and calibrate confidences (temperature scaling).
\end{itemize}
"""),

    ("AI usage", r"""
\begin{itemize}
\item \textbf{Claude (Anthropic), via Claude Code}: wrote and debugged the training, evaluation and Gradio code, the Graphviz architecture diagram and the draft of these slides.
\item All training and evaluation ran on my own laptop (Apple M4); every number comes from these runs (\texttt{results/results.csv}, \texttt{results/train.log}), none from papers, model cards or AI answers.
\item No AI tool was used to generate data or labels; the datasets are public Bitext and CLINC150.
\end{itemize}
"""),
]
