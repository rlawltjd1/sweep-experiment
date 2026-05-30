#!/bin/bash
#SBATCH --job-name=steer_sweep
#SBATCH --gres=gpu:1                 # GPU 개수
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00              # walltime — 죽어도 resume 되니 짧게 잡고 재제출 가능
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
# #SBATCH --partition=gpu            # 클러스터에 맞게 주석 해제/수정
# #SBATCH --account=YOUR_ACCOUNT

set -e
mkdir -p logs results

# --- 환경 (택1) ---
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate steer
# source .venv/bin/activate

# --- HF 토큰 (Llama / HateXplain gated) ---
export HF_TOKEN=hf_xxx
export HF_HOME=$SCRATCH/hf_cache

# --- 1주차 산출물 precheck ---
need="results/probe.pkl results/b0_baseline.json \
results/fn_subset_eval_v1.npy results/fn_subset_eval_toxigen_v1.npy \
src/eval/metrics.py data/eval/eval_v1.csv data/eval/eval_toxigen_v1.csv \
cell_c_test_final.csv cell_bbb_domain_v10_256_revised.csv"
for f in $need; do
    [ -f "$f" ] || { echo "[ERROR] 누락: $f  — week1 산출물/입력을 업로드하세요"; exit 1; }
done

# --- 2주차 sweep (벡터는 run_sweep이 cell csv + HateXplain로 생성) ---
srun python week2_sweep.py --batch 64 --all-layers --eval both

# 죽으면 같은 스크립트 재제출 → 끝난 셋업 건너뛰고 이어서 돈다