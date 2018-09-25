#!/bin/bash
set -o pipefail
set -e

abspath() { echo $(readlink -e $1); }

entry=$1
model=$2
src=$3
ref_stem=$4

spm_dir=$(dirname $(abspath $src))
spm_model=$spm_dir/spm.model

beamsize=10
bleu_script=/home/pluiefox/repos/mosesdecoder/scripts/generic/multi-bleu.perl

translate="python $entry translate --model $model --beamsize $beamsize --normalize --quiet"
calc_bleu="perl $bleu_script -lc $ref_stem"
parse_zh_script="/home/pluiefox/ai_challenger_translation_train_20170912/code/scripts/val-parse-zh.py"
reverse_script="/home/pluiefox/ai_challenger_translation_train_20170912/code/scripts/reverse.sh"
decode_spm="spm_decode --model=$spm_model"

COLOR='\033[0;32m'
NOCOLOR='\033[0m'

echo -e validation script: "${COLOR}$translate < $src 2>/dev/null | $reverse_script | $decode_spm | $parse_zh_script | $calc_bleu | cut -f 3 -d ' ' | cut -f 1 -d ','${NOCOLOR}" >&2
bleu=$($translate < $src | $reverse_script | $decode_spm | $parse_zh_script | $calc_bleu | cut -f 3 -d ' ' | cut -f 1 -d ',')

echo $bleu
