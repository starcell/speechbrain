
1) KdialectSpeech data prepare for classification

dialectclassifier_prepare.py를 실행하여 데이터 파일을 준비 :
토크나이저의 결과 csv를 이용, 모든 방언에 대하여 Tokenizer를 실행하여 결과 csv 파일을 준비

dialectclassifier_prepare.py 실행결과 dataset(csv) 파일이 생성됨
    train.csv
    valid.csv
    test.csv


train_ecapa.yaml에 설정

province_codes: 분류할 방언 종류
    ['gw', 'gs', 'jl', 'jj', 'cc']


num_province_data: 샘플링할 데이터 수 : 실제 데이터 보다 적어야 한다.(가장 수가 적은 데이터 보다 적게)
    train: 14000
    valid: 1700
    test: 1700


sentence_len: 여기에 설정된 길이(초) 만큼만 음성데이터를 사용(기본 3초, 5, 7 테스트)


out_n_neurons: 분류할 방어의 수
    모든 방언을 다 하면 5
    데이터가 적은 제주를 빼고하면 4
