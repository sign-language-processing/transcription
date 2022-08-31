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
  --user_defined_symbols='$SW$,$HNS$,$de$,$az$,$hi$,$dsl$,$lv$,$ro$,$kbh$,$ak$,$psr$,$nsl$,$af$,$vn$,$zh-tw$,$bg$,$eu$,$cb$,$sn$,$mx$,$za$,$kr$,$nh$,$po$,$di$,$sh$,$sgg$,$gv$,$th$,$kk$,$bfi$,$rm$,$hu$,$al$,$eg$,$gsg$,$np$,$he$,$sv$,$kb$,$gd$,$pe$,$es$,$cs$,$jk$,$cz$,$nl$,$ve$,$xh$,$tmh$,$pck$,$hds$,$tsq$,$ji$,$my$,$bn$,$eo$,$uy$,$gb$,$it$,$br$,$da$,$chr$,$gr$,$ru$,$py$,$sr$,$oj$,$cj$,$swl$,$ca$,$do$,$pp$,$cq$,$fsl$,$ise$,$ssp$,$is$,$gj$,$et$,$vi$,$cn$,$vgt$,$zu$,$tu$,$ss$,$pot$,$csl$,$ka$,$qc$,$uk$,$bzs$,$ac$,$ml$,$am$,$sk$,$lt$,$el$,$dk$,$mfs$,$pt$,$gss$,$jp$,$sy$,$ko$,$dj$,$en$,$tw$,$fil$,$jsl$,$ie$,$ja$,$ncs$,$mr$,$sa$,$mi$,$cp$,$mk$,$up$,$so$,$ag$,$fi$,$ni$,$pl$,$se$,$ph$,$isg$,$sq$,$tl$,$sg$,$us$,$mm$,$bs$,$ck$,$tr$,$cl$,$sfb$,$no$,$fa$,$te$,$co$,$ar$,$qu$,$mt$,$wa$,$fr$,$ew$,$id$,$ase$,$tn$,$be$,$hy$,$sl$,$wo$,$ke$,$ch$,$gn$,$zh$,$hn$' \
  --model_prefix="${vocab_dir}/vocab" --vocab_size=16000 \
  --input="${vocab_dir}/data.src.txt,${vocab_dir}/data.trg.txt" \
  --input_sentence_size="${sample_size}" --shuffle_input_sentence=true

rm "${vocab_dir}/data.src.txt" "${vocab_dir}/data.trg.txt"

mv "${vocab_dir}/vocab.model" "${vocab_output}"

