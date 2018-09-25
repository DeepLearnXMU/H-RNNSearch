#!/bin/bash
export THEANO_FLAGS=device=$1

a=(2 3 4 6 8)
for i in ${a[@]}
do    
    python rnnsearch.py translate --model $2.best.pkl --normalize --test /home/lemon/data/zh-en/d80/punc/test/nist0${i}.src /home/lemon/data/zh-en/d80/punc/test/nist0${i}.index >$2nist0${i}.txt 2>$2test0${i}.out
done
cat $2nist02.txt $2nist03.txt $2nist04.txt $2nist06.txt \
$2nist08.txt > $2nist_all.txt
