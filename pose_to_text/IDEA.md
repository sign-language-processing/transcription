# Automatic Transcription of Sign Languages Using SignWriting

## Introduction

Sign languages are rich and diverse forms of communication, yet their transcription and documentation face challenges
due to the lack of standardized writing systems. SignWriting, a unique notation system, seeks to universally represent
sign languages in written form. This research proposal aims to develop an automatic transcription system for sign
languages using SignWriting notation. The system will take a video of a single sign in sign language as input and
generate SignWriting as output.

## Literature Review

### SignWriting

Valerie Sutton introduced SignWriting in 1974 as a visual script designed to represent sign languages. This script
captures the movements, facial expressions, and body positions unique to each sign. SignWriting has found applications
in education, research, and daily communication within the deaf community.

## Sign Language Datasets

1. Sign2MINT: A lexicon of German Sign Language (DGS) that focuses on natural science subjects, featuring 5,263 videos
   with SignWriting transcriptions.
2. SignSuisss: A Swiss Sign Language Lexicon that covers all three Swiss sign languages: Swiss-German Sign Language (
   DSGS), Langue des Signes Française (LSF), and Lingua Italiana dei Segni (LIS). Approximately 5,000 LSF videos include
   SignWriting transcriptions in SignBank.

## Methodology

The proposed research will utilize the Neural Machine Translation (NMT) framework to model the problem as a
sequence-to-sequence task, using JoeyNMT 2.2.0 for experimentation.

### Data Representation

1. Input: Videos will be preprocessed with the MediaPipe framework to extract 3D skeletal poses. While these poses have
   limitations and do not capture the full range of a sign, they serve as a compromise for avoiding video input.
2. Output: Formal SignWriting in ASCII (FSW), e.g., "M518x529S14c20481x471S27106503x489"

### Data Preprocessing

Outputs will be tokenized to separate shape, orientation, and position, e.g., "M 518 x 529 S14c 2 0 481 x 471 S271 0 6
503 x 489". Predictable symbols, such as "M" and "x," will be removed to create a more compact sequence: "518 529 S14c 2
0 481 471 S271 0 6 503 489".

### Experiments

#### Input Experiments

- Use poses as they are.
- Pose Sequence Length Reduction: As the input pose sequence length often exceeds the output length, poses unchanged
  from the previous frame will be removed. Optical flow calculations and threshold checks will be used for this purpose.
- Hand and Face Normalization: To emphasize the importance of hand and face shapes, 3D hand and face normalization will
  be included in the experiments. The face will be replaced with a normalized face, and 3D normalized hands will be
  added alongside the original hands.

#### Output Experiments

1. Embedding-based token representation: At every step of the decoder, we represent input tokens using an embedding matrix.
2. Image-based token representation: Instead of inputting embeddings for each token in the decoder, the sequence up to
   that point will be drawn into an image. This method employs the same number of tokens but represents them as images
   instead of embeddings.
3. Token prediction without history: The self-attention in the decoder will be removed, and images will be "colored"
   based on the next predicted token. The decoder will receive the input sequence and the last image, predicting a token
   without considering the prediction history. This approach may improve robustness and memory efficiency, as there is
   no need to review the prediction history. During training, all images will be produced, while only the final image
   will be generated during testing.

## Conclusion

The proposed research seeks to develop an automatic transcription system for sign languages using SignWriting notation.
By leveraging the NMT framework and conducting various input and output preprocessing experiments, this project will
contribute to the advancement of sign language transcription technology. The outcomes of the experiments will be
compared and analyzed to determine the most effective method for automatic transcription using SignWriting.


```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}

\begin{document}
	
\begin{table}[h!]
    \centering
    \begin{tabular}{@{}llccc@{}}
        \toprule
        \multirow{2}{*}{\textbf{Input Experiment}} & \multirow{2}{*}{\textbf{Output Experiment}} & \multicolumn{3}{c}{\textbf{Results}} \\
        \cmidrule(lr){3-5}
        & & \textbf{Metric 1} & \textbf{Metric 2} & \textbf{Metric 3} \\
        \midrule
        \multirow{3}{*}{As is} & Embedding-based & @@ & @@ & @@ \\
        & Image-based & @@ & @@ & @@ \\
        & No history & @@ & @@ & @@ \\
        \midrule
        \multirow{3}{*}{+ Length Reduction} & Embedding-based & @@ & @@ & @@ \\
        & Image-based & @@ & @@ & @@ \\
        & No history & @@ & @@ & @@ \\
        \midrule
        \multirow{3}{*}{+ Normalization} & Embedding-based & @@ & @@ & @@ \\
        & Image-based & @@ & @@ & @@ \\
        & No history & @@ & @@ & @@ \\
        \midrule
        \multirow{3}{*}{\makecell{+ Length Reduction\\+ Normalization}} & Embedding-based & @@ & @@ & @@ \\
        & Image-based & @@ & @@ & @@ \\
        & No history & @@ & @@ & @@ \\
        \bottomrule
    \end{tabular}
    \caption{Results of Input and Output Experiments}
    \label{table:results}
\end{table}
	
\end{document}
```