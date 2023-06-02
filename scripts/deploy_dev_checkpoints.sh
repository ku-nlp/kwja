#!/usr/bin/env bash

set -euxo pipefail

usage() {
  cat << _EOT_
Usage:
  $0 remote_host remote_path

Description:
  Train and deploy checkpoints for development.
_EOT_
  exit 1
}

find_latest_checkpoint() {
  fd --extension ckpt --full-path . "$1" | sort -r | head -n 1
  return 0
}

if [[ $# -ne 2 ]]; then
  usage
fi

for task in typo senter char seq2seq word; do
  train_extra_args=("ignore_hparams_on_save=true" "trainer=cpu.debug" "do_predict_after_train=false")
  module="${task}"
  checkpoint_dir="result/${module}_module.debug-ignore_hparams_on_save_true-trainer_cpu.debug"

  poetry run python scripts/train.py -cn ${module}_module.debug "${train_extra_args[@]}"

  checkpoint_path="$(find_latest_checkpoint "${checkpoint_dir}")"
  if [[ -z "${checkpoint_path}" ]]; then
    echo "Checkpoint not found in ${checkpoint_dir}" >&2
    exit 1
  fi

  if [[ "${module}" == "typo" ]]; then
    pretrained_model_name="deberta-v2-tiny-wwm"
  elif [[ "${module}" == "senter" ]]; then
    pretrained_model_name="deberta-v2-base-wwm"
  elif [[ "${module}" == "char" ]]; then
    pretrained_model_name="deberta-v2-tiny-wwm"
  elif [[ "${module}" == "seq2seq" ]]; then
    pretrained_model_name="mt5-small"
  elif [[ "${module}" == "word" ]]; then
    pretrained_model_name="deberta-v2-tiny"
  else
    echo "Unknown module: ${module}" >&2
    exit 1
  fi
  rsync -ahPmv "${checkpoint_path}" "$1:$2/dev/${task}_${pretrained_model_name}.ckpt"
done
