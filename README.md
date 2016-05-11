# char_rnn_ari
한글 Character RNN 구현

# 0. 파일 설명
## Hangulpy/Hangulpy.py
https://github.com/rhobot/Hangulpy에서folk한 한글 관련 분해/조합 코드
오토마타, 문장 분리 코드 추가

## char_rnn_test.ipynb
테스트용 코드 (내부에 Hangulpy.py 로딩처리 추가)

## conv2utf8.py
수집한 텍스트 파일 인코딩이 제각각인 관계로 utf8로 통합 변환하는코드.
10KB 로딩 후 인코딩 추측, 컨버팅 해서 저장 하는 코드
data/origin/*.txt를 모두 처리 후 data/conv/*.txt로 저장
인코딩 추측을 위해서 chardet 라이브러리에 의존
인코딩 추측을 실패한 경우 EUC_KR로 처리 후 확장자에 ".unknown" 추가
추측 확률이 0.8 이하일 경우 확장자에 ".notsure"추가 

## deparse.py
형태소 조합 테스트를 위해서 배열에 있는 형태소 묶음을 조합해서 출력하는 코드
Hangulpy의존

## parse.py
input.txt 파일을 형태소로 분리해주는 코드 /data/conv/*.txt 모두 읽어서 분리후 출력
Hangulpy의존

## train.py
data_dir에 지정된(han1~han5중 1) 폴더를 읽어서 트레이닝 처리

# 1. 타이핑된 책 텍스트 수동으로 모음
data/origin/*.txt 로 저장 ( 2처리 후 삭제 했음)

# 2. 인코딩 타입 utf8로 통일
python conv2utf8.py
data/conv/*.txt 로 컨버팅됨. ( 3처리 후 conv.zip으로 압축 처리했음)

# 3. 형태소 분리
python parse.py > data/index.txt
data/conv/*.txt를 모두 읽어서 형태소 분리, 한글+라틴 외에 삭제 후 출력하는 코드
stdout으로 출력하는 형태로 파일 저장을 위해서
linux shell의 output redirection을 써서 data/index.txt으로 저장.

# 4. 파일 나누기
split -l 2500000 index.txt index.
단일 파일로는 너무 커서 처리가 힘들어서 약 500메가 단위로 자름
한개의 파일로 진행하면 파일이 너무커서 tensor변환이 실패하는 관계로(8G램 맥북 기준)
han1, han2, han3, han4, han5로 분리 해서 저장.

# 5. 트레이닝
python train.py
data_dir 폴더에 지정된 폴더 내용을 읽어서 트레이닝
han1~han5 중 사용하는 한개 폴더 외에 주석 처리 상태

* input이 변경되면 단어장을 새로 만드는 형태여서 연결해서 트레이닝을 위해선 수정 필요.

# 6. 출력 내용 복원 (* 노트북의 test사용 시는 조합 된 형태로 출력 됨)
deparse.py 수정
=> data 변수에 출력된 문자 배열 저장 후 실행
