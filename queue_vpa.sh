#!/bin/bash
# Coupled brackets with hidden types (vpa.py): plain networks of 4, 2, 6 and 3 layers, same settings otherwise.
#   bash queue_vpa.sh [config]
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
cd "$(dirname "$0")"
mkdir -p logs
config=${1:-vpa_shift}
for L in 4 2 6 3; do
  name=${config}_plain_d128_L${L}_T128_s0
  echo "$(date '+%m-%d %H:%M') start $name" >> logs/queue_vpa.log
  $PY -W ignore vpa_train.py --config $config --layers $L --device mps > logs/$name.train.log 2>&1 || echo "  TRAIN FAILED $name" >> logs/queue_vpa.log
  echo "$(date '+%m-%d %H:%M') done  $name: $(tail -1 logs/$name.train.log | cut -c1-120)" >> logs/queue_vpa.log
done
echo "$(date '+%m-%d %H:%M') queue finished" >> logs/queue_vpa.log
