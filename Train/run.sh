#!/usr/bin/env -S --default-signal=INT bash
cd "$(dirname "$0")/.."
exec >> train.log 2>&1
export NCCL_PROTO=Simple
export HIPBLASLT_TENSILE_LIBPATH=$PWD/hipblaslt-lib/gfx1201
echo $$ > train.pid
stop=0
child=
trap 'stop=1; [ -n "$child" ] && kill -INT "$child" 2>/dev/null' INT TERM
(
    while kill -0 $$ 2>/dev/null; do
        line=$(date +%F_%T)
        for c in 0 1; do
            d=/sys/class/drm/card$c/device
            h=$(echo $d/hwmon/hwmon*)
            line+=" | card$c $(cat $h/temp1_input $h/temp2_input $h/temp3_input $h/power1_average $h/fan1_input $d/gpu_busy_percent 2>/dev/null | tr '\n' ' ')"
        done
        echo "$line"
        if [ $(($(date +%s) - $(stat -c %Y train.log))) -gt 180 ]; then
            Train/hang.sh $$ >> train.log 2>&1
        fi
        sleep 30
    done
) >> gpu.log 2>&1 &
args=("$@")
run=
before=0
fails=0
while [ "$stop" = 0 ]; do
    echo "[run.sh] $(date +%F_%T) start ${args[*]}"
    started=$(date +%s)
    .venv/bin/torchrun --standalone --nproc_per_node 2 "${SCRIPT:-Train/train.py}" "${args[@]}" &
    child=$!
    wait "$child"
    rc=$?
    while kill -0 "$child" 2>/dev/null; do
        wait "$child"
        rc=$?
    done
    echo "[run.sh] $(date +%F_%T) exit rc=$rc stop=$stop"
    [ "$stop" = 1 ] && break
    [ "$rc" = 0 ] && break
    [ -f "$run/latest.pt" ] || run=$(grep '^run ' train.log | tail -n 1 | cut -d' ' -f2)
    after=0
    if [ -f "$run/latest.pt" ]; then
        after=$(stat -c %Y "$run/latest.pt")
        args=(--resume "$run")
    fi
    if [ "$after" = "$before" ]; then
        fails=$((fails + 1))
    else
        fails=0
    fi
    before=$after
    echo "[run.sh] $(date +%F_%T) crash rc=$rc no_progress=$fails next ${args[*]}"
    if [ "$fails" -ge 3 ]; then
        echo "[run.sh] $(date +%F_%T) GIVE UP"
        exit 1
    fi
    sleep 30
    if journalctl -k --since "@$started" --no-pager 2>/dev/null | grep -qE 'GPU reset end with ret = -|resume of IP block .* failed|device lost from bus|PSP resume failed'; then
        echo "[run.sh] $(date +%F_%T) GPU reset failed, reboot needed, GIVE UP"
        exit 1
    fi
done
echo "[run.sh] $(date +%F_%T) END"
