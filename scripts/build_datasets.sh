#!/usr/bin/zsh

while getopts a:w:s:j:o: OPT
do
  case $OPT in
    a) ACTIVATOR="$OPTARG" ;;
    w) WORK_DIR="$OPTARG" ;;
    s) SCRIPTS="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    *)
      echo "invalid option"
      exit ;;
  esac
done

# shellcheck source=/dev/null
source "$ACTIVATOR"
mkdir -p "$WORK_DIR" "$OUT_DIR"/{kyoto,kwdlc,fuman}
git clone git@github.com:ku-nlp/KyotoCorpusFull.git "$WORK_DIR"/KyotoCorpusFull
git clone git@github.com:ku-nlp/KWDLC.git "$WORK_DIR"/KWDLC
git clone git@github.com:ku-nlp/AnnotatedFKCCorpus.git "$WORK_DIR"/AnnotatedFKCCorpus
python "$SCRIPTS"/add_features_to_raw_corpus.py "$WORK_DIR"/KyotoCorpusFull/knp "$WORK_DIR"/kyoto/knp --ne-tags /mnt/zamia/kodama/jula/data/kyoto_ne/CRL.input -j "$JOBS"
python "$SCRIPTS"/add_features_to_raw_corpus.py "$WORK_DIR"/KWDLC/knp "$WORK_DIR"/kwdlc/knp -j "$JOBS"
python "$SCRIPTS"/add_features_to_raw_corpus.py "$WORK_DIR"/AnnotatedFKCCorpus/knp "$WORK_DIR"/fuman/knp -j "$JOBS"
kyoto idsplit \
  --corpus-dir "$WORK_DIR"/kyoto/knp \
  --output-dir "$OUT_DIR"/kyoto \
  --train "$WORK_DIR"/KyotoCorpusFull/id/full/train.id \
  --valid "$WORK_DIR"/KyotoCorpusFull/id/full/dev.id \
  --test "$WORK_DIR"/KyotoCorpusFull/id/full/test.id
kyoto idsplit \
  --corpus-dir "$WORK_DIR"/kwdlc/knp \
  --output-dir "$OUT_DIR"/kwdlc \
  --train "$WORK_DIR"/KWDLC/id/split_for_pas/train.id \
  --valid "$WORK_DIR"/KWDLC/id/split_for_pas/dev.id \
  --test "$WORK_DIR"/KWDLC/id/split_for_pas/test.id
kyoto idsplit \
  --corpus-dir "$WORK_DIR"/fuman/knp \
  --output-dir "$OUT_DIR"/fuman \
  --train "$WORK_DIR"/AnnotatedFKCCorpus/id/split_for_pas/train.id \
  --valid "$WORK_DIR"/AnnotatedFKCCorpus/id/split_for_pas/dev.id \
  --test "$WORK_DIR"/AnnotatedFKCCorpus/id/split_for_pas/test.id
