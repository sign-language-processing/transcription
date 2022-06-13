# Pose-to-Text

Pose to text model, for text generation from a sign language pose sequence.

## Main Idea

An autoregressive seq2seq model, encoding poses.

#### Pseudo code:

```python
pose_embedding = embed(pose)
previous_tokens = ["<SOS>"]
while True:
    next_token = decode(pose_embedding, previous_tokens)
    previous_tokens.append(next_token)
    yield next_token
```