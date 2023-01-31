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