#!/bin/bash
std=
no=

a=(2 3 4 6 8)
for i in ${a[@]}
do
    perl multi-bleu.perl -lc /home/lemon/data/zh-en/d/test/nist0${i}.tok.ref < $1nist0${i}.txt.rmunk
done
perl multi-bleu.perl -lc /home/lemon/data/zh-en/d/test/nist_all.tok.ref < $1nist_all.txt.rmunk
