#!/bin/bash
# Mess3 with the nesting depth given in the input (linearly), same settings and seed as night_d128_L4_T128_s1;
# then the unseen-depth test and the step diagnosis on the result.
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
cd "$(dirname "$0")"
mkdir -p logs
name=depthin_d128_L4_T128_s1
$PY -W ignore train.py --process mess3 --layers 4 --d 128 --seed 1 --steps 20000 --depth_input --tag depthin --device mps > logs/$name.train.log 2>&1 || { echo "TRAIN FAILED"; exit 1; }
$PY -W ignore depth_ood.py runs/$name > runs/$name.depth.txt 2>&1 || echo "DEPTH TEST FAILED"
$PY -W ignore depth_steps.py runs/$name > runs/$name.depth_steps.txt 2>&1 || echo "STEP DIAGNOSIS FAILED"
echo "done $name"
