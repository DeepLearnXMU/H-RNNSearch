#!/bin/bash

set -e

src=$1
l2r_nbest=$2
model_r2l=$3

python scripts/reverse_nbest.py < $l2r_nbest | python rnnsearch.py rescore --model $model_r2l --source $src --normalize | python scripts/rerank.py | scripts/reverse.sh

