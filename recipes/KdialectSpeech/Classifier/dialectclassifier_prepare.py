#!/usr/bin/env python
"""Recipe for prepare KdialectSpeech Dataset.

To run this recipe, use the following command:
> python train.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:

    hparams/train_epaca.yaml (for the ecapa+tdnn system)

Author
    * N Park 2022
    * @Starcell
"""

import pandas as pd
import os
import sys
import logging

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)

def skip(hparams):
    skip = True
    if not os.path.exists(hparams['train_csv']):
        skip = False

    if not os.path.exists(hparams['valid_csv']):
        skip = False

    if not os.path.exists(hparams['test_csv']):
        skip = False
    
    return skip


def make_dialect_df(hparams):

    province_codes = hparams['province_codes']
    dialect_data_folder = hparams['dialect_data_folder']
    num_province_data = hparams['num_province_data']
    seed = hparams['seed']


    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for province_code in province_codes:
        print(f'province_code : {province_code}')
        province_dir = os.path.join(dialect_data_folder, province_code)
        if not os.path.isdir(province_dir):
            print(f'{province_dir} is not exist!')
            continue
        else:
            for tvt in ['train', 'valid', 'test']:
                # logger.info(f'tvt : {tvt}')
                print(f'tvt : {tvt}')
                csv = os.path.join(province_dir, tvt + '.csv')
                csv_df = pd.read_csv(csv)

                csv_df_sampled = csv_df.sample(n=num_province_data[tvt], random_state=seed)

                if tvt == 'valid':
                    valid_df = pd.concat([valid_df, csv_df_sampled])
                elif tvt == 'test':
                    test_df = pd.concat([test_df, csv_df_sampled])
                else:
                    train_df = pd.concat([train_df, csv_df_sampled])
    
    return (train_df, valid_df, test_df)

if __name__ == "__main__":

    logger.info("Starting preparing...")

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    if skip(hparams):
        logger.info('csv files are prepared aleady!')
    else:
        logger.info('csv files are not exist, starting make csv files!')

        (train_df, valid_df, test_df) = make_dialect_df(hparams)

        train_csv = hparams['train_csv']
        valid_csv = hparams['valid_csv']
        test_csv = hparams['test_csv']

        os.makedirs(os.path.dirname(train_csv), exist_ok=True)
        train_df.to_csv(train_csv)
        os.makedirs(os.path.dirname(valid_csv), exist_ok=True)
        valid_df.to_csv(valid_csv)
        os.makedirs(os.path.dirname(test_csv), exist_ok=True)
        test_df.to_csv(test_csv)
