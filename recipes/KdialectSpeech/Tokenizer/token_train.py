#!/usr/bin/env python
"""매니패스트 파일을 입력으로 토큰화
모든 지역(강원, 경상, 충청, 전라, 제주)에 대하여 각각 토큰화
결과는 : results/<province_code> 에 저장, 예) ./results/gs
사용법 : python token_train.py
"""
import os
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece

data_dir = "/data/aidata/sentence"
province_code_list = ["gw", "gs", "cc", "jl", "jj"]
model_type = "unigram"
vocab_size = 5000

for prinvince in province_code_list:
    data_file = os.path.join(data_dir, prinvince + ".csv")
    model_dir = os.path.join("results", prinvince)

    sb.create_experiment_directory(
        experiment_directory=model_dir,
        hyperparams_to_save=None,
        overrides={},
    )

    spm = SentencePiece(
        model_dir=model_dir,
        vocab_size=vocab_size,
        annotation_train=data_file,
        annotation_read="wrd",
        model_type=model_type,
        character_coverage=1.0,
        bos_id=1,
        eos_id=2
    )

