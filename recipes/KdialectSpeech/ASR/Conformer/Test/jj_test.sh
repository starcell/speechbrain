#!/usr/bin/env bash
set -x
python test_data_prepare.py jj
# nohup python test_swer.py --device cuda:0 test_jj.yaml &> nohup_jj.out &

exit 0