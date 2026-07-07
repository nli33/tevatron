#!/usr/bin/env bash
#SBATCH --job-name=nanochat-bidi-screen
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=H200
#SBATCH -o JOB-%x-%j.out
#SBATCH -e JOB-%x-%j-err.out
#
# Short-schedule screen for the bidirectional-attention change, in isolation.
#
# Trains a nanochat retriever for a small, self-consistent step budget on
# MS MARCO (schedule re-scaled to the short horizon: short warmup + linear
# decay to 0), then evaluates on two tiny BEIR corpora (SciFact, NFCorpus).
# Everything is identical between the two configs except the causal vs
# bidirectional attention mask, so the delta isolates that one change.
#
# Submit both arms:
#   sbatch --export=ALL,TAG=causal,BIDIR=0 scripts/nanochat_bidirectional_screen.sbatch
#   sbatch --export=ALL,TAG=bidi,BIDIR=1   scripts/nanochat_bidirectional_screen.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

set +u
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate new
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PWD}/src"
PY=/u2/n262li/.conda/envs/new/bin/python

TAG="${TAG:-causal}"
BIDIR="${BIDIR:-0}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

MODEL=/u2/n262li/models/karpathy-nanochat-d32-hf
CACHE=/u2/n262li/tevatron-runs/nanochat-msmarco/hf-datasets
RUN=/u2/n262li/tevatron-runs/nanochat-bidi-screen/${TAG}
ADAPTER=${RUN}/retriever
mkdir -p "${ADAPTER}" "${RUN}/logs" "${RUN}/beir"

echo "[INFO] TAG=${TAG} BIDIR=${BIDIR} MAX_STEPS=${MAX_STEPS} on $(hostname); code=${PWD}/src"
nvidia-smi -L || true

BIDI_FLAG=()
[[ "${BIDIR}" == "1" ]] && BIDI_FLAG=(--bidirectional)

# ---------------- 1. short training ----------------
echo "[INFO] ===== train (${TAG}) ====="
"${PY}" -m tevatron.retriever.driver.train \
  --output_dir "${ADAPTER}" \
  --model_name_or_path "${MODEL}" \
  --pooling eos --append_eos_token --padding_side left --normalize \
  --attn_implementation sdpa --bf16 "${BIDI_FLAG[@]}" \
  --lora --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --dataset_name SamanthaZJQ/msmarco-passage-aug-2.0 --dataset_split train \
  --corpus_name SamanthaZJQ/msmarco-passage-corpus-2.0 --corpus_split train \
  --dataset_cache_dir "${CACHE}" \
  --query_prefix 'Query: ' --passage_prefix 'Passage: ' \
  --query_max_len 32 --passage_max_len 156 \
  --train_group_size 16 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --gradient_checkpointing \
  --learning_rate 1e-4 --temperature 0.02 \
  --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}" --lr_scheduler_type linear \
  --logging_steps 10 --save_steps "${MAX_STEPS}" --save_total_limit 1 \
  2>&1 | tee "${RUN}/logs/train.${TAG}.log"

# ---------------- 2. eval on small BEIR ----------------
eval_ds() {
  local ds="$1"
  local dd="${RUN}/beir/${ds}"
  local emb="${dd}/embeddings" rk="${dd}/runs" lg="${dd}/logs"
  mkdir -p "${emb}" "${rk}" "${lg}"
  echo "[INFO] ===== encode+search+eval ${ds} (${TAG}) ====="

  "${PY}" -m tevatron.retriever.driver.encode \
    --output_dir "${dd}" --model_name_or_path "${MODEL}" \
    --pooling eos --append_eos_token --padding_side left --normalize \
    --attn_implementation sdpa "${BIDI_FLAG[@]}" \
    --lora_name_or_path "${ADAPTER}" \
    --dataset_name Tevatron/beir-corpus --dataset_config "${ds}" --dataset_split train \
    --dataset_cache_dir "${CACHE}" \
    --passage_prefix 'Passage: ' --passage_max_len 512 \
    --per_device_eval_batch_size 64 \
    --encode_output_path "${emb}/corpus.${ds}.pkl" 2>&1 | tee "${lg}/encode-corpus.log"

  "${PY}" -m tevatron.retriever.driver.encode \
    --output_dir "${dd}" --model_name_or_path "${MODEL}" \
    --pooling eos --append_eos_token --padding_side left --normalize \
    --attn_implementation sdpa "${BIDI_FLAG[@]}" \
    --lora_name_or_path "${ADAPTER}" \
    --dataset_name nli33/beir --dataset_config "${ds}" --dataset_split test \
    --dataset_cache_dir "${CACHE}" --encode_is_query \
    --query_prefix 'Query: ' --query_max_len 512 \
    --per_device_eval_batch_size 64 \
    --encode_output_path "${emb}/queries.${ds}.pkl" 2>&1 | tee "${lg}/encode-queries.log"

  # Force CPU FAISS: the installed faiss-gpu binary has no kernel for the H200
  # (CUDA error 209 "no kernel image available"), so hide the GPU for search.
  CUDA_VISIBLE_DEVICES="" "${PY}" -m tevatron.retriever.driver.search \
    --query_reps "${emb}/queries.${ds}.pkl" \
    --passage_reps "${emb}/corpus.${ds}.pkl" \
    --depth 100 --batch_size 64 --save_text \
    --save_ranking_to "${rk}/rank.${ds}.txt" 2>&1 | tee "${lg}/search.log"

  "${PY}" -m tevatron.utils.format.convert_result_to_trec \
    --input "${rk}/rank.${ds}.txt" --output "${rk}/rank.${ds}.trec" --remove_query

  "${PY}" -m pyserini.eval.trec_eval -c -mndcg_cut.10 -mrecall.100 \
    "beir-v1.0.0-${ds}-test" "${rk}/rank.${ds}.trec" 2>&1 | tee "${rk}/metrics.${ds}.txt"
}

for ds in scifact nfcorpus; do eval_ds "${ds}"; done

echo "[SCREEN DONE] ${TAG}"
grep -H "ndcg_cut_10\|recall_100" "${RUN}"/beir/*/runs/metrics.*.txt || true
