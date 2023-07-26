#!/usr/bin/env bash
set -x
python test_data_prepare.py gs
nohup python test_swer.py --device cuda:1 test_gs.yaml &> nohup_gs.out &

exit 0