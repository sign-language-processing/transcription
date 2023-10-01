# Surprising Applications of Video-to-Text in Sign Language Translation

## Introduction:

Sign Language, a vital mode of communication for the deaf has for too long been inadequately served by the limited available translation methods. 
This research proposal aims to explore innovative, 
technology-aided improvements in the translation between signed languages, represented in SignWriting, and spoken languages. 
Our study diverges from the traditional approach and proposes to utilize a more visually-focused technique which mirrors the human process of reading SignWriting as a sequence of images rather than a conventional text.

Previously, Jiang et al. (2023) laid down a significant foundation in this field, with their work on translating SignWriting using specialized parsing and factorized machine translation. 
Building upon this work, we strive to integrate SignWriting into large pre-trained language models in a more natural and holistic manner. 
We propose a comparison between using a text-based SignWriting encoder to an image-based encoder that leverages Convolutional Neural Networks (CNNs) or a Vision Transformer (ViT). 
The image-based encoders treat each sign as either a single token (in the case of CNNs) or as 32 tokens (in the case of ViT), offering a radically different approach to encoding SignWriting.

To accomplish this, we will utilize BLIP-2 (Li et al. 2023) and, based on VideoBLIP (Yu et al. 2023), we will encode a sequence of images as an input. 
To decode, we will employ the advanced and highly effective encoder-decoder-based LLM, FLAN-T5-XL, by fine-tuning Salesforce/blip2-flan-t5-xl.

We assert that image-based encoding of SignWriting provides a more effective and versatile method for translating sign languages when compared to traditional text-based encoding. 
Through this research, we aim to provide compelling evidence to substantiate our theory, which, if confirmed, could pave the way for a significant breakthrough in sign language translation technologies.

## Related Work:

Jiang et al. (2023) explore text-to-text sign to spoken language translation, with SignWriting as the chosen sign language notation system. Despite SignWriting usually represented in 2D, they use the 1D Formal SignWriting specification and propose a neural factored machine translation approach to encode sequences of the SignWriting graphemes as well as their position in the 2D space. They verify the proposed approach on the SignBank dataset in both a bilingual setup (American Sign Language to English) and two multilingual setups (4 and 21 signed-to-spoken language pairs, respectively). They apply several low-resource machine translation techniques used to improve spoken language translation to similarly improve the performance of sign language translation. Their findings validate the use of an intermediate text representation for signed language translation, and pave the way for including sign language translation in natural language processing research.

Yu et al. (2023) propose VideoBLIP, a video-to-text translation model that leverages the BLIP-2 architecture. 
They encode a sequence of images using the BLIP image encoder, and flatten them into a single sequence of image patch tokens.
Trained end-to-end, their model can generate texts based on videos.

Li et al. (2023) propose BLIP-2, a new architecture for image-to-text translation.
They propose a generic and efficient pretraining strategy that bootstraps vision-language pre-training from 
off-the-shelf frozen pre-trained image encoders and frozen large language models.

The Bergamot project developed pipelines for fast, local, multilingual machine translation models.
Based on marian, they developed a pipeline for training and quantizing models.
This only supports text-to-text translation, and expects a shared source-target vocabulary, and a huge amount of data.

## Methodology:

Our approach to SignWriting translation involves a novel paradigm that interprets SignWriting as a series of images, akin to the human reading process. 
The methodology employed to bring this concept to fruition is detailed as follows:
By following these steps, our research aims to shed light on the unexpected and untapped potential of video-to-text and image-based methodologies in sign language translation.


### Dataset Preparation: 
We will utilize the SignBank dataset, which includes a diverse array of signed languages represented in SignWriting. 
As per the requirements of our model, the dataset will be prepared by converting the 1D Formal SignWriting specification into 2D images.

In addition to the SignBank dataset, we have undertaken a manual data collection effort to further strengthen our model. 
We have annotated fingerspelling letters and numbers in 22^[American, Brazilian, British, Chinese, Danish, Flemish, French, French Belgian, German, Honduran, Irish, Israeli, Italian, Japanese, Mexican, Nicaraguan, Norwegian, Portuguese, Spanish, Swedish, Swiss German, and Thai.] different signed languages. 
The fingerspelling is mostly taken from: https://www.signwriting.org/forums/software/fingkeys/fkey001.html
The transcribed was mostly performed by Sutthikhun Phaengphongsai, paid by `sign.mt ltd`.
To make our model robust to fingerspelling, we have artificially generated 10K words from wordslist^[https://github.com/imsky/wordlists]
and 4K numbers sampled from 0 to 10^9.
and numbers in the aforementioned signed languages deterministically.

### Dataset cleaning
We note that the SignBank dataset contains many issues, and is not immidiately fit for machine translation
It includes SignWriting entries with text that is not parallel, or multiple terms where only some of them are parallel
(for example, it includes a chapter and page number for a book, but not the text, or a word and its definition).

We manually correct at least 5 entries per puddle. Some puddles are somewhat formulaic, and we can correct many entries at once using rules.
In the appendix, we include the rules we used.

Then, we use ChatGPT on all of the text entries, and implement two pseudo-functions:

`clean(#number-of-signs, language-code, terms)` that takes the number of signs, language code, and existing terms, and returns a clean version of the terms that are parallel.
For example, `clean(1, "sl", ["Koreja (mednarodno)", "Korea"])` returns `["Koreja", "Korea"]`,
And `clean(18, "en", ["Acts 04_27-31c", "James Orlow"])` returns `[]`

`expand(language-code, terms)` that takes the language code and clean terms, and expands the terms to include paraphrases and correct capitalization.
Since some of the terms in the data are in English, we ask the function to return both the language and english separately.
For example, `expand("sv", ["tre"])` returns `{"sv": ["Tre", "3"], "en": ["Three"]}`
And `expand("de", ["Vater", "father"])` returns `{"de": ["Vater", "Vati", "Papa", "Erzeuger"], "en": ["Father", "Dad", "Daddy"]}`


### Modelling Selection: 
We will employ two different types of image-based encoders - CNNs and ViTs. 
For the CNN-based approach, each sign will be treated as a single token, while the ViT-based approach will treat each sign as 32 tokens. 
Both encoders will be compared to a baseline model that uses a text-based SignWriting encoder.

The image sequence generated will be encoded as input using the BLIP-2 (Li et al. 2023) model, leveraging its efficient vision-language pretraining strategy.
We will further implement VideoBLIP's methodology (Yu et al. 2023), which allows us to flatten the encoded image sequence into a single sequence of image patch tokens.

The encoded sequence will be fed into FLAN-T5-XL, a large language model fine-tuned based on Salesforce/blip2-flan-t5-xl. 
This model will act as our decoder and generate the translated text in the target spoken language.

Finally, we also attempt a bergamot pipeline, which is a text-to-text translation, and the signwriting is encoded as text.

### Evaluation:
The proposed models' performance will be evaluated by comparing the translated text to the ground truth in the target language using chrF scores. 
Comparative analysis between the models will be done to determine the efficacy of image-based encoding over traditional text-based encoding.
Finally we compare to previous work (Jiang et al. 2023) on the SignBank dataset, using their public API.

