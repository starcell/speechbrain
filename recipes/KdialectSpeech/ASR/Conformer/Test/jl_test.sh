#!/usr/bin/env bash
set -x
python test_data_prepare.py jl
nohup python test_swer.py --device cuda:3 test_jl.yaml &> nohup_jl.out &

exit 0