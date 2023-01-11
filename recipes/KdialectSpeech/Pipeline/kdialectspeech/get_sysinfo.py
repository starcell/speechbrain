#!/usr/bin/env python

'''
실행법 : python run_pipe.py run_pipe.yaml

speechbrain/utils/superpowers.py 사용 검토
'''

import logging
import sys
import os
import subprocess
import glob
import shutil

# from tqdm import tqdm
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml



