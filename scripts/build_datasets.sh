#!/usr/bin/env bash

JOBS=1

usage() {
  echo "Usage: scripts/build_dataset.sh --work-dir=\${WORK_DIR} --out-dir=\${OUT_DIR} --jobs=${JOBS}"
  echo "*** NOTE: specify arguments with \"=\" ***"
  echo "  --work-dir    path to working directory"
  echo "  --out-dir     path to output directory"
  echo "  --jobs        number of jobs (default=1)"
}

while getopts h-: opt; do
  if [[ $opt = "-" ]]; then
    opt=`echo ${OPTARG} | awk -F "=" '{print $1}'`
		OPTARG=`echo ${OPTARG} | awk -F "=" '{print $2}'`
  fi

  case "$opt" in
    work-dir)
      WORK_DIR=$OPTARG
      ;;
    out-dir)
      OUT_DIR=$OPTARG
      ;;
    jobs)
      JOBS=$OPTARG
      ;;
    h|help)
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

if [[ -z "$WORK_DIR" ]] || [[ -z "$OUT_DIR" ]]; then
  echo "missing required arguments"
  usage
  exit 1
fi

mkdir -p "$WORK_DIR" "$OUT_DIR"/{kwdlc,fuman}
git clone git@github.com:ku-nlp/KWDLC.git "$WORK_DIR"/KWDLC
git clone git@github.com:ku-nlp/AnnotatedFKCCorpus.git "$WORK_DIR"/AnnotatedFKCCorpus
poetry run python ./scripts/build_dataset.py "$WORK_DIR"/KWDLC/knp "$OUT_DIR"/kwdlc \
  --id "$WORK_DIR"/KWDLC/id/split_for_pas \
  -j "$JOBS"
poetry run python ./scripts/build_dataset.py "$WORK_DIR"/AnnotatedFKCCorpus/knp "$OUT_DIR"/fuman \
  --id "$WORK_DIR"/AnnotatedFKCCorpus/id/split_for_pas \
  -j "$JOBS"
