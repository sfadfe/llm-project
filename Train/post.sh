#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export HIPBLASLT_TENSILE_LIBPATH=$PWD/hipblaslt-lib/gfx1201
ok() { grep '^\[run.sh\] .* exit rc=' train.log | tail -n 1 | grep -q 'rc=0 stop=0'; }

SCRIPT=Train/sft.py Train/run.sh "$@"
ok || { echo "[post.sh] $(date +%F_%T) SFT not finished, stop" >> train.log; exit 1; }
sft=$(grep '^run .*/hyb200m_sft_' train.log | tail -n 1 | cut -d' ' -f2)
echo "[post.sh] $(date +%F_%T) SFT done $sft" >> train.log
HIP_VISIBLE_DEVICES=0 .venv/bin/python Train/chat.py "$sft" >> train.log 2>&1

SCRIPT=Train/rl.py Train/run.sh --init "$sft"
ok || { echo "[post.sh] $(date +%F_%T) RL not finished, stop" >> train.log; exit 1; }
rl=$(grep '^run .*/hyb200m_rl_' train.log | tail -n 1 | cut -d' ' -f2)
echo "[post.sh] $(date +%F_%T) RL done $rl" >> train.log
HIP_VISIBLE_DEVICES=0 .venv/bin/python Train/chat.py "$rl" >> train.log 2>&1
echo "[post.sh] $(date +%F_%T) ALL DONE" >> train.log
