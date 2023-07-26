#!/usr/bin/env bash
set -x
python test_data_prepare.py cc
nohup ./test_swer.py --device cuda:2 test_cc.yaml &> nohup_cc.out &

exit 0