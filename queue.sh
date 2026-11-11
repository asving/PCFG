#!/bin/bash
# Overnight queue: train, then validate, then draw figures, for a list of runs on one device.
#   bash queue.sh DEVICE "process layers d seed steps" ["process layers d seed steps" ...]
# Everything is logged under logs/; a failed step does not stop the queue.
cd "$(dirname "$0")" || exit 1
PY=../group-belief/.venv/bin/python
DEV=$1; shift
mkdir -p logs
for spec in "$@"; do
  set -- $spec
  proc=$1; L=$2; d=$3; seed=$4; steps=$5
  name="night_d${d}_L${L}_T128_s${seed}"
  [ "$proc" != "mess3" ] && name="${proc}_${name}"
  echo "$(date '+%m-%d %H:%M') start $name on $DEV" >> logs/queue_$DEV.log
  extra=""; [ "$DEV" = "cpu" ] && extra="--threads 4"
  $PY -W ignore train.py --process "$proc" --layers "$L" --d "$d" --seed "$seed" --steps "$steps" \
      --device "$DEV" --tag night $extra > "logs/$name.train.log" 2>&1 || echo "  TRAIN FAILED $name" >> logs/queue_$DEV.log
  $PY -W ignore validate.py "runs/$name" > "runs/$name.validate.txt" 2>&1 || echo "  VALIDATE FAILED $name" >> logs/queue_$DEV.log
  $PY -W ignore figures.py "runs/$name" "figures/$name" > "logs/$name.figures.log" 2>&1 || echo "  FIGURES FAILED $name" >> logs/queue_$DEV.log
  echo "$(date '+%m-%d %H:%M') done  $name: $(grep 'THE NETWORK' "runs/$name.validate.txt" | tr -s ' ' | cut -c1-60)" >> logs/queue_$DEV.log
done
echo "$(date '+%m-%d %H:%M') queue finished" >> logs/queue_$DEV.log
