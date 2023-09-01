#!/usr/bin/env bash

JOBS=1

usage() {
  cat << _EOT_
Usage:
  ./scripts/build_dataset.sh --out-dir=<OUT_DIR> [--jobs=<JOBS>]
  *** NOTE: specify arguments with "=" ***

Options:
  --out-dir     path to output directory
  --jobs        number of jobs (default=1)
_EOT_
}

while getopts h-: opt; do
  if [[ $opt = "-" ]]; then
    opt=$(echo "${OPTARG}" | awk -F "=" '{print $1}')
    OPTARG=$(echo "${OPTARG}" | awk -F "=" '{print $2}')
  fi

  case "$opt" in
  out-dir)
    OUT_DIR=$OPTARG
    ;;
  jobs)
    JOBS=$OPTARG
    ;;
  h | help)
    usage
    exit 0
    ;;
  *)
    echo "invalid option -- $opt"
    usage
    exit 1
    ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  echo "missing required arguments"
  usage
  exit 1
fi

WORK_DIR="$(mktemp -d)"

mkdir -p "$WORK_DIR" "$OUT_DIR"/{kwdlc,fuman,wac}
git clone --depth 1 git@github.com:ku-nlp/KWDLC.git "$WORK_DIR"/KWDLC
git clone --depth 1 git@github.com:ku-nlp/AnnotatedFKCCorpus.git "$WORK_DIR"/AnnotatedFKCCorpus
git clone --depth 1 git@github.com:ku-nlp/WikipediaAnnotatedCorpus.git "$WORK_DIR"/WikipediaAnnotatedCorpus
poetry run python ./scripts/build_dataset.py "$WORK_DIR"/KWDLC/knp "$OUT_DIR"/kwdlc \
  --id "$WORK_DIR"/KWDLC/id/split_for_pas \
  --doc-id-format kwdlc \
  -j "$JOBS"
poetry run python ./scripts/build_dataset.py "$WORK_DIR"/AnnotatedFKCCorpus/knp "$OUT_DIR"/fuman \
  --id "$WORK_DIR"/AnnotatedFKCCorpus/id/split_for_pas \
  -j "$JOBS"
poetry run python ./scripts/build_dataset.py "$WORK_DIR"/WikipediaAnnotatedCorpus/knp "$OUT_DIR"/wac \
  --id "$WORK_DIR"/WikipediaAnnotatedCorpus/id \
  --doc-id-format wac \
  -j "$JOBS"

rm -rf "$WORK_DIR"
