#!/bin/bash
export THEANO_FLAGS=device=gpu0
set -o pipefail
set -e

entry=$1
model=$2
src=$3
isrc=$4
ref_stem=$5

beamsize=10
bpe=false
bleu_script=/home/pluiefox/repos/mosesdecoder/scripts/generic/multi-bleu.perl

translate="python $entry translate --model $model --test $src $isrc --beamsize $beamsize --normalize"
restore_bpe="sed -r 's/(@@ )|(@@ ?$)//g'"
calc_bleu="perl $bleu_script -lc $ref_stem"

if [[ $bpe == "true" ]]; then
    bleu=$($translate | sed -r 's/(@@ )|(@@ ?$)//g' | $calc_bleu | cut -f 3 -d ' ' | cut -f 1 -d ',')
else
    bleu=$($translate | $calc_bleu | cut -f 3 -d ' ' | cut -f 1 -d ',')
fi

echo $bleu
