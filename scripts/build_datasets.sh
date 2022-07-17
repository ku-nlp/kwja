#!/usr/bin/zsh

while getopts a:w:s:j:o: OPT
do
  case $OPT in
    a) ACTIVATOR="$OPTARG" ;;
    w) WORK_DIR="$OPTARG" ;;
    s) SCRIPTS="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    *) echo "invalid option";;
  esac
done

# shellcheck source=/dev/null
source "$ACTIVATOR"
mkdir -p "$WORK_DIR" "$OUT_DIR"/{kwdlc,fuman}
cd "$WORK_DIR" || return
git clone git@github.com:ku-nlp/KyotoCorpusFull.git
git clone git@github.com:ku-nlp/KWDLC.git
git clone git@github.com:ku-nlp/AnnotatedFKCCorpus.git
python "$SCRIPTS"/add_features.py KyotoCorpusFull/knp kyoto/knp -j "$JOBS"
python "$SCRIPTS"/add_features.py KWDLC/knp kwdlc/knp -j "$JOBS"
python "$SCRIPTS"/add_features.py AnnotatedFKCCorpus/knp fuman/knp -j "$JOBS"
mkdir -p kyoto/split
kyoto idsplit \
  --corpus-dir kyoto/knp \
  --output-dir kyoto/split \
  --train KyotoCorpusFull/id/full/train.id \
  --valid KyotoCorpusFull/id/full/dev.id \
  --test KyotoCorpusFull/id/full/test.id
kyoto idsplit \
  --corpus-dir kwdlc/knp \
  --output-dir "$OUT_DIR"/kwdlc \
  --train KWDLC/id/split_for_pas/train.id \
  --valid KWDLC/id/split_for_pas/dev.id \
  --test KWDLC/id/split_for_pas/test.id
kyoto idsplit \
  --corpus-dir fuman/knp \
  --output-dir "$OUT_DIR"/fuman \
  --train AnnotatedFKCCorpus/id/split_for_pas/train.id \
  --valid AnnotatedFKCCorpus/id/split_for_pas/dev.id \
  --test AnnotatedFKCCorpus/id/split_for_pas/test.id
python "$SCRIPTS"/split_document.py kyoto/split "$OUT_DIR"/kyoto
