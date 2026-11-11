#!/bin/bash
# Continue the overnight plan after an interruption such as a restart.
#   bash resume.sh          relaunch whatever is not finished
#   bash resume.sh --dry    only list what would be launched
# A run counts as finished when its validation file is complete; a run that was cut off starts again.
cd "$(dirname "$0")" || exit 1
PY=../group-belief/.venv/bin/python
GPU=("tomq 6 128 0 20000" "clock3 4 128 0 20000" "mess3+open 4 128 0 20000" "fern 4 128 0 20000" "tomq 4 128 0 20000"
     "mess3 4 128 1 20000" "clock3 4 128 1 20000" "tomq 6 128 1 20000" "tomq+open 6 128 0 20000")
CPU=("mess3 4 32 0 20000" "mess3 4 16 0 20000" "mess3 4 64 0 20000" "mess3 4 32 1 20000" "mess3 4 16 1 20000"
     "mess3 4 64 1 20000" "clock3 4 32 0 20000")
finished() {  # process layers d seed
  local name="night_d${3}_L${2}_T128_s${4}"
  [ "$1" != "mess3" ] && name="${1}_${name}"
  [ "$(grep -c -E "tokens changed|content changed" "runs/$name.validate.txt" 2>/dev/null)" -ge 3 ]
}
todo_gpu=(); for s in "${GPU[@]}"; do finished $s || todo_gpu+=("$s"); done
todo_cpu=(); for s in "${CPU[@]}"; do finished $s || todo_cpu+=("$s"); done
todo_coupled=(); for L in 4 2 3 6; do
  grep -q '"step": 20000' "runs/coupled0.9_night_d128_L${L}_T128_s0.json" 2>/dev/null || todo_coupled+=("$L"); done
echo "GPU queue:     ${#todo_gpu[@]} to run: ${todo_gpu[*]}"
echo "CPU queue:     ${#todo_cpu[@]} to run: ${todo_cpu[*]}"
echo "coupled queue: ${#todo_coupled[@]} to run: layers ${todo_coupled[*]}"
[ "$1" = "--dry" ] && exit 0
if pgrep -f "queue.sh|coupled_train.py|train.py" > /dev/null; then echo "queues are still running; nothing launched"; exit 1; fi
mkdir -p logs
[ ${#todo_gpu[@]} -gt 0 ] && nohup bash queue.sh mps "${todo_gpu[@]}" >> logs/queue_mps.out 2>&1 &
[ ${#todo_cpu[@]} -gt 0 ] && nohup bash queue.sh cpu "${todo_cpu[@]}" >> logs/queue_cpu.out 2>&1 &
if [ ${#todo_coupled[@]} -gt 0 ]; then
  nohup bash -c '
    sleep 5; while pgrep -f "queue.sh mps" > /dev/null; do sleep 60; done
    for L in '"${todo_coupled[*]}"'; do
      echo "$(date "+%m-%d %H:%M") start coupled L$L (resumed)" >> logs/queue_coupled.log
      '"$PY"' -W ignore coupled_train.py --layers $L --device mps > "logs/coupled_L$L.train.log" 2>&1
      echo "$(date "+%m-%d %H:%M") done  coupled L$L: $(tail -1 "logs/coupled_L$L.train.log" | tr -s " " | cut -c1-70)" >> logs/queue_coupled.log
    done' >> logs/queue_coupled.out 2>&1 &
fi
echo "launched; follow logs/queue_*.log"
