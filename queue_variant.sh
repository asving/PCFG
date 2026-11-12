#!/bin/bash
# One Mess3 network with part of the bracket structure handed to it, same settings and seed as
# night_d128_L4_T128_s1; then the unseen-depth test and the diagnoses on the result.
#   bash queue_variant.sh tagin --tag_input [wait_for_this_file]
#   bash queue_variant.sh deprot --depth_rotary [wait_for_this_file]
#   bash queue_variant.sh rotboth "--depth_rotary --pos_rotary"
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
cd "$(dirname "$0")"
mkdir -p logs
tag=$1; flag=$2; wait_for=$3
if [ -n "$wait_for" ]; then while [ ! -f "$wait_for" ]; do sleep 15; done; fi
name=${tag}_d128_L4_T128_s1
$PY -W ignore train.py --process mess3 --layers 4 --d 128 --seed 1 --steps 20000 $flag --tag $tag --device mps > logs/$name.train.log 2>&1 || { echo "TRAIN FAILED $name"; touch logs/$tag.done; exit 1; }
touch logs/$tag.done
$PY -W ignore depth_ood.py runs/$name > runs/$name.depth.txt 2>&1 || echo "DEPTH TEST FAILED"
$PY -W ignore depth_steps.py runs/$name > runs/$name.depth_steps.txt 2>&1 || echo "STEP DIAGNOSIS FAILED"
$PY -W ignore depth_alias.py runs/$name > runs/$name.depth_alias.txt 2>&1 || echo "ANGLE DIAGNOSIS FAILED"
if [[ "$flag" == *pos_rotary* ]]; then      # no position table, so the network can be run past its training length
  $PY -W ignore length_ood.py runs/$name 512 > runs/$name.length.txt 2>&1 || echo "LENGTH TEST FAILED"
fi
echo "done $name"
