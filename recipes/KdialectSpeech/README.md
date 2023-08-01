추가 설치 패키지

ffmpeg : ASR trainning에서 없으면 경고 나옴.
apt update
apt install ffmpeg

pydub : 음성 파일을 분할할 때 사용(ffmpeg를 필요로함)
pip install pydub

boto3 : 클라우드 오브젝트 스토리지에서 데이터 다운로드 받을 때 필요
pip install boto3

sb v0.5.13 기반으로 starcell branch에서 수정
sb 수정 사항

sb.utils.metric_stats.py
    sWER%로 수정

inference 수정
    bug fix : inference 안되는 문제 해결


## 유효성 검증 자료 만들기   

### dockcer export, import 사용

1) 컨테이너를 tar file로 export 하기
    docker export [옵션] 컨테이너명|컨테이너ID > export받을 파일
    ex)
```bash
docker export kdialect-test-0726 > kdialectspeech-test.tar
```
이렇게 익스포트한 파일을 복사해서 사용 > gw_gs.tar, cc_jl_jj.tar
보관룔 압축 : kdialectspeech-test.tar -> kdialectspeech-test.tar.zip

2) 컨테이너 export file을 import해서 사용하기
    import 방법(아래 두 가지 방법 중 하나)
    $docker import <exported file> <container image name>
    또는
    $cat <exported file>  | docker import - <container image name>
    ex)

```bash
docker import kdialectspeech-test.tar kdialectspeech-test:20230727
```
위와 아래 예시는 동일한 내용
```bash
docker kdialectspeech-test.tar : docker import -  kdialectspeech-test:20230727
```

3) 도커 컨테이너 실행
    docker run -itd \
    --name <container name> \
    --gpus all \
    --ipc=host \
    -v /etc/localtime:/etc/localtime:ro \
    -v <your-data-dir>:/data  \
    <container image name> /usr/bin/bash &

ex)
```bash
docker run -itd \
--name kdialectspeech-test \
--gpus all \
--ipc=host \
-v /etc/localtime:/etc/localtime:ro \
-v /home/starcell/data:/data  \
kdialectspeech-test:20230727 /usr/bin/bash &
```

4) docker login : attach를 사용해도 됨, attach를 사용하면 사용 후 컨테이너 자동 종료됨
docker exec -it <container name> bash
```bash
docker exec -it kdialectspeech-test bash
```


### 아래는 참고 : 테스트용
도커 이미지 만들기 
    불필요한 파일들을 지우고 이미지 커밋
    docker commit <container name> <image name:tag>

ex)
```bash
docker commit kdialect-test kdialect-test:20230726
```


도커 실행
    docker run -itd \
    --name kdialect-test-0726 \
    --gpus all \
    --ipc=host \
    -v /etc/localtime:/etc/localtime:ro \
    -v <your-data-dir>:/data 

ex)
```bash
docker run -itd \
--name kdialect-test-0726 \
--gpus all \
--ipc=host \
-v /etc/localtime:/etc/localtime:ro \
-v /home/starcell/data:/data  \
kdialect-test:20230726 /usr/bin/bash &
```
