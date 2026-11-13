#!/usr/bin/env bash
sup=$1
tr=$(ps -o pid=,comm= --ppid "$sup" | awk '$2!="bash" && $2!="sleep"{print $1}')
[ -n "$tr" ] || exit 0
workers=$(ps -o pid= --ppid "$tr")
[ -n "$workers" ] || exit 0
dir=hang/$(date +%F_%H%M%S)
mkdir -p "$dir"
echo "[run.sh] $(date +%F_%T) hang log silent capture $dir kill" $workers
for w in $workers; do
    kill -USR1 "$w"
    {
        echo "=== worker $w"
        for t in /proc/$w/task/*; do
            echo "$(basename "$t") $(cat "$t/comm") $(awk '{print $3}' "$t/stat") $(cat "$t/wchan")"
        done
        echo "evicted_ms $(cat /sys/class/kfd/kfd/proc/$w/stats_*/evicted_ms | tr '\n' ' ')"
    } >> "$dir/proc.txt" 2>&1
done
for c in 0 1; do
    d=/sys/class/drm/card$c/device
    {
        echo "=== card$c busy $(cat $d/gpu_busy_percent) mem_busy $(cat $d/mem_busy_percent)"
        cat $d/pp_dpm_sclk $d/pp_dpm_mclk
    } >> "$dir/proc.txt" 2>&1
done
for f in kfd/mqds kfd/hqds dri/0000:03:00.0/amdgpu_fence_info dri/0000:07:00.0/amdgpu_fence_info; do
    timeout -k 2 15 sudo -n cat "/sys/kernel/debug/$f" > "$dir/$(echo "$f" | tr '/:' '__')" 2>&1 &
done
sleep 18
timeout -k 2 15 sudo -n python3 Train/aql.py "$dir/kfd_hqds" $workers > "$dir/aql.txt" 2>&1
kill -9 $workers
