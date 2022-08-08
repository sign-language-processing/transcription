# Transcription

Repository for sign language transcription related models.

Ideally pose based models should use a shared large-pose-language-model,
able to encode arbitrary pose sequence lengths, and pre-trained on non-autoregressive reconstruction.

- [shared](shared) - includes shared utilities for all models
- [pose_to_segments](pose_to_segments) - segments pose sequences
- [text_to_pose](text_to_pose) - animates poses using text
- [pose_to_text](pose_to_text) - generates text from poses
