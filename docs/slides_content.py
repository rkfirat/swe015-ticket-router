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
\usepackage[T1]{fontenc}
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
\item A support desk receives thousands of free-text messages a day; each must reach the right team \textbf{instantly}.
\item Users: the support team (routing) and customers (fast answer).
\item The app takes a message $\rightarrow$ normalizes it $\rightarrow$ predicts the intent with a confidence $\rightarrow$ routes it.
\item Low-confidence or out-of-scope messages go to a \textbf{human agent}; greetings and thanks get a rule-based reply.
\item Domain change (allowed by Section 4): English \textbf{CLINC150} intents (banking, travel, home, work, \dots) instead of appliances -- public, labeled, with a real out-of-scope class.
\end{itemize}
\textbf{Success} = high in-scope accuracy \emph{and} high out-of-scope recall (a wrong team is worse than a human agent), at CPU latency of a few ms.
"""),

    ("Development: data", r"""
\begin{itemize}
\item \textbf{CLINC150 ``plus''} (Larson et al., EMNLP 2019, CC BY 3.0), Hugging Face \texttt{clinc/clinc\_oos}.
\item 150 intents in 10 domains + \texttt{oos} (out-of-scope) class.
\item Fixed splits: train 15,250 $\cdot$ validation 3,100 (100 OOS) $\cdot$ \textbf{test 5,500 (1,000 OOS)}.
\item All models trained on train; confidence thresholds tuned on validation; \textbf{every number on the following slides is from the same test split}.
\item Preprocessing (identical for all models): Unicode NFKC, URL/e-mail/model-code tokens (\texttt{DW-6534}), lowercase, repeated letters (\textit{neeed} $\rightarrow$ \textit{need}), punctuation removal, edit-distance spelling correction.
\end{itemize}
"""),

    ("Development: architecture", r"""
\centering
\vspace*{\fill}\includegraphics[width=\linewidth]{architecture.png}\vspace*{\fill}
"""),

    ("Development: technique map", r"""
\small
\begin{tabular}{>{\raggedright}p{2.4cm}>{\raggedright}p{6.2cm}>{\raggedright\arraybackslash}p{4.2cm}}
\toprule
\textbf{Area} & \textbf{Technique used} & \textbf{Where in the code} \\ \midrule
A $\cdot$ Neural networks & Feedforward net on averaged word2vec, trained with SGD (momentum) + backprop & \texttt{train.py: train\_ffn}, \texttt{src/models.py: FFN} \\
B $\cdot$ Classical NLP & Regex normalization, model-code extraction, min.\ edit distance spelling correction, ELIZA-style rules, tf-idf + logistic regression; own skip-gram word2vec & \texttt{src/common.py}, \texttt{train.py: train\_tfidf} \\
C $\cdot$ Transformers & BERT-base (uncased) fine-tuned with a 151-way classification head; WordPiece tokenizer & \texttt{train.py: train\_bert} \\
D $\cdot$ Efficient models & Knowledge distillation BERT-base $\rightarrow$ BERT-mini, then dynamic INT8 quantization; size and latency benchmark & \texttt{train.py: train\_distill}, \texttt{src/models.py}, \texttt{evaluate.py} \\
\bottomrule
\end{tabular}
"""),

    ("Development: the four models and how they were trained", r"""
\small
\begin{enumerate}
\item \textbf{Classical baseline:} tf-idf (word 1--2 grams, sublinear tf) + logistic regression ($C{=}20$).
\item \textbf{Own neural network:} skip-gram word2vec (200-d, window 5, 40 epochs) trained on the training messages; averaged message vector $\rightarrow$ FFN (256 ReLU, dropout 0.3) $\rightarrow$ softmax; SGD lr 0.1, momentum 0.9, 60 epochs, best validation epoch kept.
\item \textbf{Transformer:} \texttt{bert-base-uncased} (110M params) fine-tuned 4 epochs, AdamW lr $4\cdot10^{-5}$, batch 32, linear warm-up/decay (Apple M4 GPU, MPS).
\item \textbf{Quantized Transformer:} \texttt{BERT-mini} student (4 layers, 256 hidden, 11M params) trained 25 epochs on $0.5\cdot$KL(teacher, T=2) + $0.5\cdot$CE; then PyTorch dynamic quantization: \texttt{Linear} weights INT8, embedding table 8-bit weight-only.
\end{enumerate}
Routing: if $\max p < \tau$ or the class is \texttt{oos} $\rightarrow$ human agent. $\tau$ per model maximizes (in-scope acc + OOS recall)/2 on validation.
"""),

    ("Development: results (same held-out test set, 5,500 messages)", r"""
\centering\footnotesize
@@RESULTS@@

\vspace{3mm}
\raggedright\tiny @@HARDWARE@@ Size = files on disk needed by the model. Rows 4 and 5 are ablations (quantization of the teacher; student before quantization).
"""),

    ("Conclusion: what the comparison showed", r"""
\small
\begin{itemize}
\item \textbf{BERT-base wins on quality}: 91.3\% accuracy and 92.1\% OOS recall, +7.9 points over the tf-idf baseline -- but 438 MB and 17.8 ms per message.
\item \textbf{The classical baseline is strong}: tf-idf + LogReg (83.4\%) beats my own word2vec + FFN (81.5\%). Averaging word vectors loses word order, and word2vec trained on 15k short messages is weak. The FFN is the smallest and fastest model (4.8 MB, 0.9 ms).
\item \textbf{Distillation brought the speed}: the BERT-mini student keeps 98\% of the teacher's accuracy (89.5\%) and is 9.8$\times$ smaller and 9.6$\times$ faster.
\item \textbf{Quantization brought the size}: INT8 cuts the student from 44.8 to 11.6 MB (3.9$\times$) for $-0.55$ points accuracy. On the M4 CPU, dynamic INT8 is \emph{slower} for this small model (1.85 $\rightarrow$ 2.83 ms), because quantizing activations at every layer costs more than the 256-wide matrix multiplications save. On BERT-base, INT8 is 4$\times$ smaller and 15\% faster with no loss.
\item \textbf{Final app model}: 11.6 MB, 2.8 ms, 89.0\% accuracy -- 38$\times$ smaller and 6$\times$ faster than BERT-base for $-2.4$ points.
\end{itemize}

"""),

    ("Conclusion: failure examples and limitations", r"""
\small
Final model: 607 of 5,500 test messages wrong. \textbf{399} are in-scope messages sent to a human agent (the safe error), \textbf{112} are out-of-scope messages routed to a team, and 96 went to the wrong team.
\begin{itemize}
\item \textit{``what is the official language of romania''} $\rightarrow$ \texttt{change\_language} (0.94). This is a keyword trap: BERT-base and tf-idf correctly send it to a human. The student learned ``language'' and is over-confident.
\item \textit{``how do i fix the ice maker in my fridge''} $\rightarrow$ \texttt{food\_last} (0.85). CLINC has no appliance intents, so the model picks the nearest kitchen intent.
\item \textit{``I lost my credit card, please freeze my account''} $\rightarrow$ human agent (0.63). The message has two intents, but the classifier is single-label.
\end{itemize}
\textbf{Limitations:} English only. Intents stand in for teams. The thresholds are tuned on only 100 out-of-scope validation messages. The latency is measured on one laptop CPU.

"""),

    ("Conclusion: next steps", r"""
\begin{itemize}
\item Collect Turkish messages (e.g. MASSIVE tr-TR) and fine-tune a multilingual student (BERTurk / XLM-R) for the original appliance scenario.
\item Map intents to real teams and add 50--100 hand-written domain examples (appliance faults, model codes).
\item Try static INT8 or ONNX Runtime kernels, where INT8 actually speeds up the matrix multiplications on ARM.
\item Calibrate confidences (temperature scaling) so the human-agent threshold is more reliable.
\end{itemize}
"""),

    ("AI usage", r"""
\begin{itemize}
\item \textbf{Claude (Anthropic), via Claude Code}: wrote and debugged the training, evaluation and Gradio code, the Graphviz architecture diagram and the draft of these slides.
\item All training and evaluation ran on my own laptop (Apple M4); every number comes from these runs (\texttt{results/results.csv}, \texttt{results/train.log}), none from papers, model cards or AI answers.
\item No AI tool was used to generate data or labels; the dataset is public CLINC150.
\end{itemize}
"""),
]
