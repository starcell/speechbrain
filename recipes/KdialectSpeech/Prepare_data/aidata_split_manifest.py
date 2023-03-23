"""매니패스트 파일에서 모델 학습에 사용할 데이터 목록을 추출
   3초~30초 사이의 데이터를 추출하고 학습용, 밸리데이션용, 테스트용으로 분리
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np


def extract_3_30_list(manifest_file):
    """매니페스트 파일에 있는 목록에서 음성의 길이가 3초에서 30초 사이인 데이터 목록을 추출
    """

    manifest_df = pd.read_csv(manifest_file)
    manifest_3_30_df = manifest_df[ (3.0 <= manifest_df["duration"]) & (manifest_df["duration"] < 30.0)]
    
    # 시간 합계 확인
    # print(f'시간 합계 : {manifest_3_30_df.duration.sum()/3600}')

    return manifest_3_30_df

def split_data_list(manifest_df, ratio):
    """dataframe을 받아서 train, valid, test로 분리하여 csv file에 저장
       ratio에 지정된 비율로 분리
    """

    manifest_len = len(manifest_df)

    train_df, valid_df, test_df = np.split(
        manifest_df.sample(frac=1, random_state=7774), 
        [int(ratio[0] * manifest_len), int(ratio[1] * manifest_len)]
    )

    return train_df, valid_df, test_df

def main():
    base_dir = Path("/data/aidata/sentence")
    ratio_for_province = {
        "gw" : [0.765, 0.875],
        "gs" : [0.77, 0.875],
        "cc" : [0.77, 0.875],
        "jl" : [0.78, 0.87],
        "jj" : [0.75, 0.86],
        }
    # gw : 0.765, 0.875 (76.5:11.0:12.5)
    # gs : 0.77, 0.875 (77.0:10.5:12.5)
    # cc : 0.77, 0.875 (77.0:10.5:12.5)
    # jl : 0.78, 0.87 (78.0:9.0:13.0)
    # jj : 0.75, 0.86 (75.0:11.0:14.0)

    for province, ratio in ratio_for_province:
        manifest_file = os.path.join(base_dir, province)
        train_df, valid_df, test_df = split_data_list(manifest_file, ratio)

        train_df.to_csv(os.path.join(base_dir, province + "_train.csv"), index=False)
        valid_df.to_csv(os.path.join(base_dir, province + "_valid.csv"), index=False)
        test_df.to_csv(os.path.join(base_dir, province + "_test.csv"), index=False)

if __name__ == "__main__":
    main()
