# Pose Encoder

## Current Implementation

Given the pose data, sequence mask, and optionally additional sequence, 
this model uses a transformer to encode the sequences.
It has a limited length (marked with `max_seq_size`) given that it relies on learned positional embeddings.

It might not be the best we can do.

## New Implementation: TODO

Naturally, a pose is visual. It has spatiotemporal information, with a strong bias towards neighboring frames.
Therefore, we should use a CNN to encode the pose.

Using the `FastAndUglyPoseVisualizer`, we can visualize each of the pose's components independently as an `Nx64x64` monochrome video.

![download-1](https://user-images.githubusercontent.com/5757359/188336807-ed468647-9ff2-4c51-bc9f-b642dea591f5.gif) 
![download](https://user-images.githubusercontent.com/5757359/188336809-68563180-ebbd-46ce-9f55-0614ffc04c1d.gif) 
![download-3](https://user-images.githubusercontent.com/5757359/188336805-029f6d7a-7a18-4100-9eed-7c9f781a489b.gif) 
![download-2](https://user-images.githubusercontent.com/5757359/188336804-3e583427-9f97-4657-8aba-f56437e4fd64.gif) 

The rational behind having each component as a separate video is that we want to cover as much as the video as possible, and avoid white space.
To generate a video with hands and face clearly visible, we would need very high video resolution.

While we can treat this as a `Nx64x64x4` tensor, it doesn't really makes sense to perform 4D convolutions.
Instead, we can use a 3D CNN (`Nx64x64`), with multiple layers, 
to "compress" the `64x64` frames to a `256` dimensional vector, for example, 
and finally convolve over the `Nx256x4` tensor, or concatenate to `Nx1024` vectors, and encode them with a transformer.