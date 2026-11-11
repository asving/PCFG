#!/bin/bash
# Coupled brackets (rho 0.9, close names the type) at several depths, on the GPU, after queue.sh mps has finished.
cd "$(dirname "$0")" || exit 1
PY=../group-belief/.venv/bin/python
while pgrep -f "queue.sh mps" > /dev/null; do sleep 60; done
for L in 4 2 3 6; do
  echo "$(date '+%m-%d %H:%M') start coupled L$L" >> logs/queue_coupled.log
  $PY -W ignore coupled_train.py --layers $L --device mps > "logs/coupled_L$L.train.log" 2>&1 || echo "  FAILED L$L" >> logs/queue_coupled.log
  echo "$(date '+%m-%d %H:%M') done  coupled L$L: $(tail -1 "logs/coupled_L$L.train.log" | tr -s ' ' | cut -c1-70)" >> logs/queue_coupled.log
done
echo "$(date '+%m-%d %H:%M') queue finished" >> logs/queue_coupled.log
