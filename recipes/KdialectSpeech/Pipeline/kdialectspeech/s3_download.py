#!/usr/bin/env python

from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import logging
import os
import sys
from pathlib import Path
import boto3
import datetime
from tqdm import tqdm

# print(os.path.dirname(os.path.abspath(__file__)))
# print(__file__)

##### download data from s3 storage
def get_s3_object_list(s3, bucket_name, prefix, max_keys):
    obj_list = []
    response = s3.list_objects(Bucket=bucket_name, MaxKeys=max_keys, Prefix=prefix)

    while True:
        if response.get('Contents') is not None:

            for content in response.get('Contents'):
                filename = content.get('Key')
                date_info = content.get('LastModified')
                
                obj_list.append(filename)
        
            if response.get('IsTruncated'):
                response = s3.list_objects(Bucket=bucket_name, MaxKeys=max_keys, Prefix=prefix,
                                        Marker=response.get('NextMarker'))
            else:
                break
        
        else:
            # logger.info(f'{prefix} : there is no data.')
            print(f'{prefix} : there is no data.')
            break
    
    return obj_list


def get_s3_files(s3, bucket_name, key_names, max_keys, data_save_path, error_file_log, root_folder=None):
    # error_file = 'error_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
    # error_file_path = os.path.join(data_save_path, error_file)
    print(f'root_folder : {root_folder}')
    with open(error_file_log, 'w') as lf:

        for key_name in key_names:
            print(key_name)
            print(f'----- date download start -----\n')
            # object_list = get_s3_object_list(s3, bucket_name, date, max_keys)
            # print(object_list[0])

            if root_folder is not None:
                key_name = root_folder + key_name
                idxs = list(range(1,7))
            else:
                idxs = list(range(0,6))

            print(f'key_name : {key_name}')
            print(f'idxs : {idxs}')

            # for key in tqdm(object_list):
            for key in tqdm(get_s3_object_list(s3, bucket_name, key_name, max_keys)):
                # print(Path(key).suffix)
                if Path(key).suffix in ['.json', '.wav']:
                    key_2 = key.split('/')
                    save_dir = os.path.join(data_save_path, key_2[idxs[1]], key_2[idxs[2]], key_2[idxs[3]], key_2[idxs[4]])
                    os.makedirs(save_dir, exist_ok=True)
                    save_file = os.path.join(save_dir, key_2[idxs[5]])
                    try:
                        # print(i)
                        s3.download_file(bucket_name, key, save_file)
                    except:
                        lf.write(f'{key}\n')
                        print(f'{key} is not exist.')
                        continue
                else:
                    lf.write(f'{key}\n')
                    print(f'{key} is not file')

            print(f'----- {bucket_name} download end -----\n')

        lf.write('----- end time : ' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    lf.close()

#####

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    ##### setup logging
    logger = logging.getLogger(__name__)

    log_config = hparams["log_config"]
    log_file = 's3_download_' + hparams["log_file"]

    logger_overrides = {
        "handlers": {"file_handler": {"filename": log_file}}
    }

    # setup_logging(config_path="log-config.yaml", overrides={}, default_level=logging.INFO)
    sb.utils.logger.setup_logging(log_config, logger_overrides)
    #####

    ##### download data from s3 storage
    # yaml에서 설정값 읽어오기 : 스토리지 접속 정보, 데이터 저장 위치
    service_name = hparams["service_name"]
    endpoint_url = hparams["endpoint_url"]
    region_name = hparams["region_name"]
    access_key = hparams["access_key"]
    secret_key = hparams["secret_key"]

    s3 = boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key)

    data_save_path = secret_key = hparams["data_save_path"]
    os.makedirs(data_save_path, exist_ok=True)

    bucket_name = hparams["bucket_name"]
    max_keys = hparams["max_keys"]
    key_names = hparams["key_names"]

    error_file_log = hparams["error_file_log"]
    get_s3_files(s3, bucket_name, key_names, max_keys, error_file_log)
    #####


