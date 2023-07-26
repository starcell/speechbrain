#!/usr/bin/env bash
set -x
python test_data_prepare.py gw
nohup python test_swer.py --device cuda:0 test_gw.yaml &> nohup_gw.out &

exit 0