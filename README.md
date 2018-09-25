A Hierarchy-to-Sequence Attentional Neural Machine Translation Model
=====================================================================

This codebase contains all scripts except training corpus to reproduce our results in the paper.

### Installation

The following packages are needed:

- Python >= 2.7
- numpy
- Theano >= 0.7 (and its dependencies).

### Preparation

First, preprocess your training corpus. Use BPE(byte-piar-encoding) to segment text into subword units for en-de translation. Please follow <https://github.com/rsennrich/subword-nmt> for further details.

To obtain vocabulary for training, run:

    python scripts/buildvocab.py --corpus /path/en.train --output /path/to/ch.voc3.pkl \
    --limit 32000 --groundhog
    python scripts/buildvocab.py --corpus /path/de.train --output /path/to/en.voc3.pkl \
    --limit 32000 --groundhog

And also, it's required to initialize encoder-backward decoder component with pretrained parameters in the proposed model of this work.

### Training

For Chinese-English experiment, do the following:

    bash train.sh

The training procedure continues about 1.5 days On a single Nvidia Titan x GPU.


### Evaluation

The evaluation metric we use is case-sensitive BLEU on tokenized reference. Translate the test set and restore text to the original segmentation:

    bash translate.sh
