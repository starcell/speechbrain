#!/usr/bin/env python
"""
매니패스트 파일을 입력으로 토큰화 트리이닝만 실행
"""
import os
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece

data_dir = "/workspace/speechbrain/recipes/KdialectSpeech/Prepare_data/data"
province_code = "cc" # gw, gs, jl, jj, cc
data_file = os.path.join(data_dir, province_code + ".csv")

model_type = "unigram"
model_dir_name = province_code + "_" + model_type + "_token_result"
model_dir = os.path.join("results", model_dir_name)

vocab_size = 5000

# total_file = "/workspace/speechbrain/recipes/KdialectSpeech/Prepare_data/data/total.csv"
# train_file = "/workspace/speechbrain/recipes/KdialectSpeech/Prepare_dat/data/total_train.csv"
# valid_file = "/workspace/speechbrain/recipes/KdialectSpeech/Prepare_dat/data/total_valid.csv"

sb.create_experiment_directory(
    experiment_directory=model_dir,
    hyperparams_to_save=None,
    overrides={},
)

spm = SentencePiece(
    model_dir=model_dir,
    vocab_size=vocab_size,
    annotation_train=data_file,
    # annotation_train=train_file,
    annotation_read="wrd",
    model_type=model_type,
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    # annotation_list_to_check=[train_file, valid_file]
)

