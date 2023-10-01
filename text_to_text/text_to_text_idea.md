# Principled Text-to-Text Sign Language Translation

## Introduction:

Sign Language, a vital mode of communication for the deaf has for too long been inadequately served by the limited available translation methods. 
This research proposal aims to explore innovative, 
technology-aided improvements in the translation between signed languages, represented in SignWriting, and spoken languages. 

Previously, Jiang et al. (2023) laid down a significant foundation in this field, with their work on translating SignWriting using specialized parsing and factorized machine translation. 

We extend on their work by:
1. cleaning the dataset they used, and extending it
2. going back to the basics of machine translation, and using a text-to-text approach without any factorization.

we believe that a cleaner dataset, and a more streamlined approach allows to train more types of implementations, and to deploy them more easily.

## Related Work:

Jiang et al. (2023) explore text-to-text sign to spoken language translation, with SignWriting as the chosen sign language notation system. Despite SignWriting usually represented in 2D, they use the 1D Formal SignWriting specification and propose a neural factored machine translation approach to encode sequences of the SignWriting graphemes as well as their position in the 2D space. They verify the proposed approach on the SignBank dataset in both a bilingual setup (American Sign Language to English) and two multilingual setups (4 and 21 signed-to-spoken language pairs, respectively). They apply several low-resource machine translation techniques used to improve spoken language translation to similarly improve the performance of sign language translation. Their findings validate the use of an intermediate text representation for signed language translation, and pave the way for including sign language translation in natural language processing research.

The Bergamot project developed pipelines for fast, local, multilingual machine translation models.
Based on marian, they developed a pipeline for training and quantizing models.
This only supports text-to-text translation, and expects a shared source-target vocabulary, and a huge amount of data.

## Methodology:

Our approach to SignWriting translation involves simple text-to-text translation, with no tricks.
The contribution of this work is the dataset, now cleaned and extended, and the implementation of the small model, deployed in the browser.

### Dataset Preparation: 
We will utilize the SignBank dataset, which includes a diverse set of signed languages represented in SignWriting, contributed by the community.

In addition to the SignBank dataset, we have undertaken a manual data collection effort to further strengthen our model. 
We have annotated fingerspelling letters and numbers in 22^[American, Brazilian, British, Chinese, Danish, Flemish, French, French Belgian, German, Honduran, Irish, Israeli, Italian, Japanese, Mexican, Nicaraguan, Norwegian, Portuguese, Spanish, Swedish, Swiss German, and Thai.] different signed languages. 
The fingerspelling is mostly taken from: https://www.signwriting.org/forums/software/fingkeys/fkey001.html
The transcription was mostly performed by Sutthikhun Phaengphongsai, paid by `sign.mt ltd`.
To make our model robust to fingerspelling, we have artificially generated 10K words from wordslist^[https://github.com/imsky/wordlists]
and 4K numbers sampled from 0 to 10^9. (TODO: sample ots more)
We further create data from strings of random letters, to make our model robust to random fingerspelling.

### Dataset cleaning
We note that the SignBank dataset contains many issues, and is not immidiately fit for machine translation
It includes SignWriting entries with text that is not parallel, or multiple terms where only some of them are parallel
(for example, it includes a chapter and page number for a book, but not the text, or a word and its definition).

We manually correct at least 5 entries per puddle. Some puddles are somewhat formulaic, and we can correct many entries at once using rules.
In the appendix, we include the rules we used.

Then, we use ChatGPT on all of the text entries, and implement two pseudo-functions:

#### Cleaning
`clean(#number-of-signs, language-code, terms)` that takes the number of signs, language code, and existing terms, and returns a clean version of the terms that are parallel.
For example, `clean(1, "sl", ["Koreja (mednarodno)", "Korea"])` returns `["Koreja", "Korea"]`,
And `clean(18, "en", ["Acts 04_27-31c", "James Orlow"])` returns `[]`
(full prompt is in the appendix)

To assess the quality of the cleaning per model, and best number of examples, we run gpt-3.5-turbo-0613 on the 5 examples per puddle from the cleaned data, 
and compare the results to the manually annotated data.
We evaluate using intersection over union, to account for the model returning additional terms, or missing some terms.
We always use the same 6 hand crafted 'motivating' examples, and compare it to adding 5 examples from the cleaned data of the same puddle.
For each model and number of examples, we report the average IOU over all puddles, and the cost.


#### Expansion
`expand(language-code, terms)` that takes the language code and clean terms, and expands the terms to include paraphrases and correct capitalization.
Since some of the terms in the data are in English, we ask the function to return both the language and english separately.
For example, `expand("sv", ["tre"])` returns `{"sv": ["Tre", "3"], "en": ["Three"]}`
And `expand("de", ["Vater", "father"])` returns `{"de": ["Vater", "Vati", "Papa", "Erzeuger"], "en": ["Father", "Dad", "Daddy"]}`
(full prompt is in the appendix)


### Experimentation

We train models for both directions of the translation, and for all languages at once, using language tags.

Since we only care about offline deployment, we train the bergamot model pipeline on this data.
We release the entire cleaned dataset to the public, to improve on our modeling.

We split the data into train, dev, and test sets, and train a model on the train set.
This is done on a shuffle of the data without overlap in signwriting.

We implement client and server side code to run the model in the browser or on the server.

### Evaluation

We report the chrF score on the test set.

We further perform an unfair comparison to Jiang et al. (2023), by using their public API on our test set.
It is unfair to us, since they might have seen the test set, and it is unfair to them, since we have a much larger dataset.

