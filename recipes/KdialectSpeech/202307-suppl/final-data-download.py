#!/usr/bin/env python3
"""NIA의 클라우드에서 데이터 다운로드
사용법 :
nohup python aidata-download-gw-gs.py &> nohup_download_gw-gs.out & 
"""
# https://guide.ncloud-docs.com/docs/storage-storage-8-2

import os
from pathlib import Path
import boto3
from tqdm import tqdm
import datetime
import logging

LOG_DIR = './'
# logfile = os.path.join(LOG_DIR, 'aidata-download-{:%Y%m%d}.log')
logfile = os.path.join(LOG_DIR, 'aidata-download.log')
logging.basicConfig(filename=logfile, level=logging.INFO)

logger = logging.getLogger(__name__)

filehandler = logging.FileHandler(logfile.format(datetime.datetime.now()), encoding='utf-8')
# formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
# filehandler.setFormatter(formatter)
logger.addHandler(filehandler)


service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
access_key = '5CA6DF7F42B860BC5BF6'
secret_key = '521AFE59000AEEE6130AA408FD0D12E56D9AB1D7'

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


bucket_name = 'aidata-2022-02-018'
prefix = '139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/07.보완조치/01.보완완료/02.라벨링데이터/'
# prefix = '139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/07.보완조치/01.보완완료/02.라벨링데이터/03.제주도/'
# prefix = '139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/'
# prefix = '139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/'

max_keys = 1000

save_path = '/data/nia/'
os.makedirs(save_path, exist_ok=True)

# err_files = []
error_file = 'error_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
error_file_path = os.path.join(save_path, error_file)
with open(error_file_path, 'w') as lf:

    logger.info(f"----- date download start : {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-----")

    # for key in tqdm(object_list):
    for key in tqdm(get_s3_object_list(s3, bucket_name, prefix, max_keys)):
        # print(Path(key).suffix)
        if Path(key).suffix in ['.json', '.wav']:
            save_dir = os.path.join(save_path, os.path.dirname(key))
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, os.path.basename(key))
            try:
                # print(f"bucket_name, key, save_file : {bucket_name}, {key}, {save_file}")
                s3.download_file(bucket_name, key, save_file)
            except:
                lf.write(f'{key}\n')
                print(f'{key} is not exist.')
                continue
        else:
            lf.write(f'{key}\n')
            print(f'{key} is not file')

    logger.info(f"----- date download end  : {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-----")

    lf.write('----- end time : ' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
lf.close()
