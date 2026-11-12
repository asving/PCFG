#!/bin/bash
# After queue_vpa.sh has finished: the second process (vpa_shift2), plain networks of 4 and 2 layers.
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
cd "$(dirname "$0")"
until grep -q "queue finished" logs/queue_vpa.log 2>/dev/null; do sleep 20; done
for L in 4 2; do
  name=vpa_shift2_plain_d128_L${L}_T128_s0
  echo "$(date '+%m-%d %H:%M') start $name" >> logs/queue_vpa2.log
  $PY -W ignore vpa_train.py --config vpa_shift2 --layers $L --device mps > logs/$name.train.log 2>&1 || echo "  TRAIN FAILED $name" >> logs/queue_vpa2.log
  echo "$(date '+%m-%d %H:%M') done  $name: $(tail -1 logs/$name.train.log | cut -c1-120)" >> logs/queue_vpa2.log
done
echo "$(date '+%m-%d %H:%M') queue finished" >> logs/queue_vpa2.log
