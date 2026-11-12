#!/bin/bash
# Unseen-depth test on each hidden-type network (about 2.5 minutes each, CPU).
PY=/Users/asvin/astera-claude/group-belief/.venv/bin/python
cd "$(dirname "$0")"
for r in night_d128_L4_T128_s1 clock3_night_d128_L4_T128_s0 clock3_night_d128_L4_T128_s1 tomq_night_d128_L6_T128_s1 fern_night_d128_L4_T128_s0 night_d64_L4_T128_s0; do
  $PY -W ignore depth_ood.py runs/$r > runs/$r.depth.txt 2>&1 || echo "FAILED $r"
  echo "done $r"
done
