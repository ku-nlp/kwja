#!/usr/bin/env bash

DEVICE=[0]
CHAR_BATCH_SIZE=1
WORD_BATCH_SIZE=1

usage() {
  echo "Usage: ./scripts/benchmark.sh --input=\${INPUT} --work-dir=\${WORK_DIR} --char-module=\${CHAR_MODULE} --word-module=\${WORD_MODULE} --device=${DEVICE} --char-batch-size=${CHAR_BATCH_SIZE} --word-batch-size=${WORD_BATCH_SIZE}"
  echo "*** NOTE: specify arguments with \"=\" ***"
  echo "  --input              path to input text"
  echo "  --work-dir           path to working directory"
  echo "  --char-module        path to fine-tuned char module"
  echo "  --word-module        path to fine-tuned word module"
  echo "  --device             device (default=[0])"
  echo "  --char-batch-size    max_batches_per_device of char module (default=1)"
  echo "  --word-batch-size    max_batches_per_device of word module (default=1)"
}

while getopts h-: opt; do
  if [[ $opt = "-" ]]; then
    opt=$(echo "${OPTARG}" | awk -F "=" '{print $1}')
    OPTARG=$(echo "${OPTARG}" | awk -F "=" '{print $2}')
  fi

  case "$opt" in
  input)
    INPUT=$OPTARG
    ;;
  work-dir)
    WORK_DIR=$OPTARG
    ;;
  char-module)
    CHAR_MODULE=$OPTARG
    ;;
  word-module)
    WORD_MODULE=$OPTARG
    ;;
  device)
    DEVICE=$OPTARG
    ;;
  char-batch-size)
    CHAR_BATCH_SIZE=$OPTARG
    ;;
  word-batch-size)
    WORD_BATCH_SIZE=$OPTARG
    ;;
  h | help)
    usage
    exit 0
    ;;
  *)
    echo "invalid option -- $opt"
    exit 1
    ;;
  esac
done

if [[ -z "$INPUT" ]] || [[ -z "$WORK_DIR" ]] || [[ -z "$CHAR_MODULE" ]] || [[ -z "$WORD_MODULE" ]]; then
  echo "missing required arguments"
  usage
  exit 1
fi

echo "Juman++"
(time cat "$INPUT" | jumanpp > "$WORK_DIR/benchmark.jumanpp.juman") 2> "$WORK_DIR/benchmark.stderr"
grep -c '^EOS$' "$WORK_DIR/benchmark.jumanpp.juman" > "$WORK_DIR/count.txt"

echo "KWJA (char_module)"
(time cat "$INPUT" | poetry run python scripts/analyze.py module=char checkpoint_path="$CHAR_MODULE" devices="$DEVICE" max_batches_per_device="$CHAR_BATCH_SIZE" +load_only=true) 2>> "$WORK_DIR/benchmark.stderr"
(time cat "$INPUT" | poetry run python scripts/analyze.py module=char checkpoint_path="$CHAR_MODULE" devices="$DEVICE" max_batches_per_device="$CHAR_BATCH_SIZE" > "$WORK_DIR/benchmark.kwja.juman") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.kwja.juman" | cut -f -3 -d "-" | uniq | wc -l >> "$WORK_DIR/count.txt"

echo "Juman++ & KNP (固有表現認識あり)"
(time cat "$INPUT" | jumanpp | knp -tab -ne-crf > "$WORK_DIR/benchmark.ne.knp") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.ne.knp" | grep -cv "ERROR:" >> "$WORK_DIR/count.txt"

echo "Juman++ & KNP (固有表現認識・照応解析あり)"
# 解析が不安定なため、一文ずつ入力
(time (
  IFS=$'\n'
  for sentence in $(cat "$INPUT" | tr '"' '\"'); do echo "${sentence}" | jumanpp | knp -tab -ne-crf -anaphora >> "$WORK_DIR/benchmark.ne_anaphora.knp"; done
)) 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.ne_anaphora.knp" | grep -cv "ERROR:" >> "$WORK_DIR/count.txt"

echo "KWJA (word_module)"
(time poetry run python scripts/analyze.py module=word checkpoint_path="$WORD_MODULE" devices="$DEVICE" max_batches_per_device="$WORD_BATCH_SIZE" +datamodule.predict.juman_file="$WORK_DIR/benchmark.kwja.juman" +load_only=true) 2>> "$WORK_DIR/benchmark.stderr"
(time poetry run python scripts/analyze.py module=word checkpoint_path="$WORD_MODULE" devices="$DEVICE" max_batches_per_device="$WORD_BATCH_SIZE" +datamodule.predict.juman_file="$WORK_DIR/benchmark.kwja.juman" > "$WORK_DIR/benchmark.kwja.knp") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.kwja.knp" | cut -f -3 -d "-" | uniq | wc -l >> "$WORK_DIR/count.txt"

grep '^real' "$WORK_DIR/benchmark.stderr" > "$WORK_DIR/time.txt"
