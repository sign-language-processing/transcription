#!/bin/bash
##
# Train SentencePiece vocabulary model
#

set -x
set -euo pipefail

test -v MARIAN

corpus_src=$1
corpus_trg=$2
vocab_output=$3
sample_size=$4

vocab_dir=$(dirname "${vocab_output}")
mkdir -p "${vocab_dir}"

pigz -dc "${corpus_src}" >"${vocab_dir}/data.src.txt"
pigz -dc "${corpus_trg}" >"${vocab_dir}/data.trg.txt"

"${MARIAN}/spm_train" --bos_id=-1 --eos_id=0 --unk_id=1 \
  --model_type="unigram" --split_by_number=0 --split_by_unicode_script=0 \
  --user_defined_symbols='$dsl$,$ni$,$sq$,$be$,$ru$,$hi$,$gsg$,$hu$,$el$,$se$,$gb$,$ar$,$ase$,$ca$,$it$,$br$,$pl$,$da$,$mfs$,$ssp$,$en$,$fa$,$swl$,$bg$,$sv$,$ko$,$tr$,$isg$,$sgg$,$ise$,$us$,$th$,$tsq$,$ja$,$pt$,$hn$,$bfi$,$he$,$mx$,$SW$,$no$,$de$,$bzs$,$fr$,$bn$,$sk$,$lv$,$sfb$,$cn$,$csl$,$et$,$fsl$,$lt$,$nl$,$vi$,$ie$,$sr$,$mk$,$uk$,$psr$,$nsl$,$dk$,$ch$,$sl$,$ncs$,$af$,$fi$,$eo$,$vgt$,$jsl$,$cs$,$es$,$zh$,$id$,$hds$,$ro$,$kk$' \
  --model_prefix="${vocab_dir}/vocab" --vocab_size=32000 \
  --input="${vocab_dir}/data.src.txt,${vocab_dir}/data.trg.txt" \
  --input_sentence_size="${sample_size}" --shuffle_input_sentence=true

rm "${vocab_dir}/data.src.txt" "${vocab_dir}/data.trg.txt"

mv "${vocab_dir}/vocab.model" "${vocab_output}"

