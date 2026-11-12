#!/bin/bash
# Kinds of open token: the non-commuting pair (vpa_s3) against the commuting control (vpa_c3), plain networks.
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
cd "$(dirname "$0")"
for spec in "vpa_s3 4" "vpa_s3 2" "vpa_c3 4" "vpa_c3 2" "vpa_s3 6" "vpa_s3 3"; do
  set -- $spec
  name=${1}_plain_d128_L${2}_T128_s0
  echo "$(date '+%m-%d %H:%M') start $name" >> logs/queue_vpa3.log
  $PY -W ignore vpa_train.py --config $1 --layers $2 --device mps > logs/$name.train.log 2>&1 || echo "  TRAIN FAILED $name" >> logs/queue_vpa3.log
  echo "$(date '+%m-%d %H:%M') done  $name: $(tail -1 logs/$name.train.log | cut -c1-120)" >> logs/queue_vpa3.log
done
echo "$(date '+%m-%d %H:%M') queue finished" >> logs/queue_vpa3.log
