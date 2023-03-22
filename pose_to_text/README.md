# Pose-to-Text

Pose to text model, for text generation from a sign language pose sequence.

## Main Idea

An autoregressive seq2seq model, encoding poses, and decoding text. (Using JoeyNMT)

To get SignWriting to work, need to modify the model:
1. self.trg_embed should take the tokenized signwriting and draw a sequence of it, at every step
2. decoder predicts from base/rotation/number distribution, it decides.
   1. In test time, we can constrain it to predict the correct part. (set -inf / 0)

