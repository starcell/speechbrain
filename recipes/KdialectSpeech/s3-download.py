#!/usr/bin/env python

# https://guide.ncloud-docs.com/docs/storage-storage-8-2

import os
from pathlib import Path
import boto3
from tqdm import tqdm
import datetime
import logging

LOG_DIR = './'
logfile = os.path.join(LOG_DIR, 's3-download-{:%Y%m%d}.log')

logging.basicConfig(filename=logfile, level=logging.INFO)

logger = logging.getLogger(__name__)

filehandler = logging.FileHandler(logfile.format(datetime.datetime.now()), encoding='utf-8')
# formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
# filehandler.setFormatter(formatter)
logger.addHandler(filehandler)


service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
access_key = '7NYHGkci0LTg3EAqu67w'
secret_key = '01cS8pqm6q0f63cJoVsIdy3CodwBjhjPnzr113rL'

s3 = boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key)


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
            print(f'{prefix} : there is no data.')
            break
    
    return obj_list



bucket_name = 'fn-2-018'
max_keys = 1000
dates = [
    # '1114Dataset_new',
    # '1115Dataset_new', # 확인 필요, 데이터 있는데 다운로드 못함
    # '1117Dataset_new',
    # '1118Dataset',
    # '1121Dataset',
    # '1122Dataset',
    # '1123Dataset',
    # '1124Dataset',
    # '1125Dataset',
    # '1126Dataset',
    # '1127Dataset',
    # '1128Dataset',
    # '1129Dataset',
    # '1130Dataset',
    # # '1204Dataset', ### data 없음
    # '1201Dataset',
    # '1202Dataset',
    # '1203Dataset',
    # '1205Dataset',
    # '1206Dataset',
    # '1207Dataset',
    # '1208Dataset',
    # '1209Dataset',
    '1210Dataset',
    # '1211Dataset', # no data
    '1212Dataset',
    '1213Dataset'
]
# dates = ['1114Dataset']

save_path = '/data/MTDATA/fn-2-018/root'
os.makedirs(save_path, exist_ok=True)

# err_files = []
error_file = 'error_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
error_file_path = os.path.join(save_path, error_file)
with open(error_file_path, 'w') as lf:

    for date in dates:
        logger.info(date)
        logger.info(f'----- date download start -----\n')
        # object_list = get_s3_object_list(s3, bucket_name, date, max_keys)
        # print(object_list[0])

        # for key in tqdm(object_list):
        for key in tqdm(get_s3_object_list(s3, bucket_name, date, max_keys)):
            # print(Path(key).suffix)
            if Path(key).suffix in ['.json', '.wav']:
                key_2 = key.split('/')
                save_dir = os.path.join(save_path, key_2[1], key_2[2], key_2[3], key_2[4])
                os.makedirs(save_dir, exist_ok=True)
                save_file = os.path.join(save_dir, key_2[5])
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

        logger.info(f'----- date download end -----\n')

    lf.write('----- end time : ' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
lf.close()
