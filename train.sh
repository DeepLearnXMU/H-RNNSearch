#!/bin/bash
export THEANO_FLAGS=device=gpu1

nohup python -u rnnsearch.py train \
--corpus /home/lemon/data/zh-en/d80/gdfa/1/ch_block_align.ch \
/home/lemon/data/zh-en/d80/gdfa/1/en_block_align.en \
/home/lemon/data/zh-en/d80/gdfa/1/index.ch \
--vocab /home/lemon/data/zh-en/dbase80/zh.vocab.pkl /home/lemon/data/zh-en/dbase80/en.vocab.pkl \
--model ghrnn \
--embdim 620 620 \
--hidden 1000 150 1000 1000 1000 \
--initialize nmt80_pre.best.pkl \
--maxhid 500 \
--deephid 620 \
--maxpart 2 \
--alpha 5e-4 \
--norm 1.0 \
--batch 80 \
--maxepoch 5 \
--seed 1234 \
--freq 1000 \
--vfreq 1500 \
--sfreq 500 \
--sort 32 \
--validation /home/lemon/data/zh-en/d80/punc/test/nist05.src /home/lemon/data/zh-en/d80/punc/test/nist05.index \
--references /home/lemon/data/mt/nist/nist05.tok.ref0 \
/home/lemon/data/mt/nist/nist05.tok.ref1 \
/home/lemon/data/mt/nist/nist05.tok.ref2 \
/home/lemon/data/mt/nist/nist05.tok.ref3 \
--optimizer rmsprop \
--shuffle 0 \
--keep-prob 0.7 \
--limit 80 80 \
--delay-val 1 \
--ext-val-script /home/lemon/work/base/HRNN/scripts/validate-nist.sh \
>cghrnn.out 2>&1 &
