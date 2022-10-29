#!/usr/bin/zsh

while getopts w:j:o: OPT
do
  case $OPT in
    w) WORK_DIR="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    *)
      echo "invalid option"
      exit ;;
  esac
done

mkdir -p "$WORK_DIR" "$OUT_DIR"/{kwdlc,fuman}
git clone git@github.com:ku-nlp/KWDLC.git "$WORK_DIR"/KWDLC
git clone git@github.com:ku-nlp/AnnotatedFKCCorpus.git "$WORK_DIR"/AnnotatedFKCCorpus
poetry run python ./scripts/build_dataset.py "$WORK_DIR"/KWDLC/knp "$OUT_DIR"/kwdlc \
  --id "$WORK_DIR"/KWDLC/id/split_for_pas \
  -j "$JOBS"
poetry run python ./scripts/build_dataset.py "$WORK_DIR"/AnnotatedFKCCorpus/knp "$OUT_DIR"/fuman \
  --id "$WORK_DIR"/AnnotatedFKCCorpus/id/split_for_pas \
  -j "$JOBS"
