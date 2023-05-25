#!/usr/bin/env bash

DEVICE="[0]"
TYPO_BATCH_SIZE=1
SENTER_BATCH_SIZE=1
CHAR_BATCH_SIZE=1
WORD_BATCH_SIZE=1

usage() {
  cat << _EOT_
Usage:
  ./scripts/benchmark.sh
    --input=<INPUT>
    --work-dir=<WORK_DIR>
    --typo-module=<TYPO_MODULE>
    --senter-module=<SENTER_MODULE>
    --char-module=<CHAR_MODULE>
    --word-module=<WORD_MODULE>
    [--device=<DEVICE>]
    [--typo-batch-size=<TYPO_BATCH_SIZE>]
    [--senter-batch-size=<SENTER_BATCH_SIZE>]
    [--char-batch-size=<CHAR_BATCH_SIZE>]
    [--word-batch-size=<WORD_BATCH_SIZE>]
  *** NOTE: specify arguments with \"=\" ***

Options:
  --input                path to input text"
  --work-dir             path to working directory"
  --typo-module          path to fine-tuned typo module"
  --senter-module        path to fine-tuned senter module"
  --char-module          path to fine-tuned char module"
  --word-module          path to fine-tuned word module"
  --device               device (default=[0])"
  --typo-batch-size      max_batches_per_device of typo module (default=1)"
  --senter-batch-size    max_batches_per_device of senter module (default=1)"
  --char-batch-size      max_batches_per_device of char module (default=1)"
  --word-batch-size      max_batches_per_device of word module (default=1)"
_EOT_
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
  typo-module)
    TYPO_MODULE=$OPTARG
    ;;
  senter-module)
    SENTER_MODULE=$OPTARG
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
  typo-batch-size)
    TYPO_BATCH_SIZE=$OPTARG
    ;;
  senter-batch-size)
    SENTER_BATCH_SIZE=$OPTARG
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

if [[ -z "$INPUT" ]] || [[ -z "$WORK_DIR" ]] || [[ -z "$TYPO_MODULE" ]] || [[ -z "$SENTER_MODULE" ]] || [[ -z "$CHAR_MODULE" ]] || [[ -z "$WORD_MODULE" ]]; then
  echo "missing required arguments"
  usage
  exit 1
fi

echo "Juman++"
(time -p cat "$INPUT" | jumanpp > "$WORK_DIR/benchmark.jumanpp.juman") 2> "$WORK_DIR/benchmark.stderr"
grep -c '^EOS$' "$WORK_DIR/benchmark.jumanpp.juman" > "$WORK_DIR/count.txt"

echo "Juman++ & KNP (NER + Dependency parsing)"
(time -p cat "$INPUT" | jumanpp | knp -tab -ne-crf -dpnd > "$WORK_DIR/benchmark.knp_ne_dpnd.knp") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.knp_ne_dpnd.knp" | grep -cv "ERROR:" >> "$WORK_DIR/count.txt"

echo "Juman++ & KNP (NER + Dependency parsing + Case analysis)"
(time -p cat "$INPUT" | jumanpp | knp -tab -ne-crf > "$WORK_DIR/benchmark.knp_ne.knp") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.knp_ne.knp" | grep -cv "ERROR:" >> "$WORK_DIR/count.txt"

echo "Juman++ & KNP (NER + Dependency parsing + PAS analysis)"
# 解析が不安定なため、一文ずつ入力
(time -p (
  IFS=$'\n'
  for sentence in $(cat "$INPUT" | tr '"' '\"'); do echo "${sentence}" | jumanpp | knp -tab -ne-crf -anaphora >> "$WORK_DIR/benchmark.knp_ne_anaphora.knp"; done
)) 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.knp_ne_anaphora.knp" | grep -cv "ERROR:" >> "$WORK_DIR/count.txt"

echo "KWJA (typo_module)"
(time -p cat "$INPUT" | poetry run python ./scripts/analyze.py module=typo checkpoint_path="$TYPO_MODULE" devices="$DEVICE" max_batches_per_device="$TYPO_BATCH_SIZE" > "$WORK_DIR/benchmark.kwja.txt") 2>> "$WORK_DIR/benchmark.stderr"

echo "KWJA (senter_module)"
(time -p cat "$INPUT" | poetry run python ./scripts/analyze.py module=senter checkpoint_path="$SENTER_MODULE" devices="$DEVICE" max_batches_per_device="$SENTER_BATCH_SIZE" > "$WORK_DIR/benchmark.kwja.senter") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.kwja.senter" | cut -f -3 -d "-" | uniq | wc -l >> "$WORK_DIR/count.txt"

echo "KWJA (char_module)"
(time -p poetry run python ./scripts/analyze.py module=char checkpoint_path="$CHAR_MODULE" devices="$DEVICE" max_batches_per_device="$CHAR_BATCH_SIZE" +datamodule.predict.senter_file="$WORK_DIR/benchmark.kwja.senter" > "$WORK_DIR/benchmark.kwja.juman") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.kwja.juman" | cut -f -3 -d "-" | uniq | wc -l >> "$WORK_DIR/count.txt"

echo "KWJA (word_module)"
(time -p poetry run python ./scripts/analyze.py module=word checkpoint_path="$WORD_MODULE" devices="$DEVICE" max_batches_per_device="$WORD_BATCH_SIZE" +datamodule.predict.juman_file="$WORK_DIR/benchmark.kwja.juman" > "$WORK_DIR/benchmark.kwja.knp") 2>> "$WORK_DIR/benchmark.stderr"
grep "# S-ID:" "$WORK_DIR/benchmark.kwja.knp" | cut -f -3 -d "-" | uniq | wc -l >> "$WORK_DIR/count.txt"

grep '^real' "$WORK_DIR/benchmark.stderr" > "$WORK_DIR/time.txt"
