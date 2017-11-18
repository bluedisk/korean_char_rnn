# korean char rnn
Implementation of Korean Character Recussive Nueral Network   
한글 Character RNN 구현

한 글자의 음소 분리 후 모양새는 다음과 같습니다.
[초성] [중성] [종성] [음절구문문자] 
ex: 안녕하세요? => ㅇㅏㄴᴥㄴㅕㅇᴥㅎㅏᴥㅅㅔᴥㅇㅛᴥ?
(음절 구분 문자는 단지 글자가 귀여워서  ᴥ 를 사용 했습니다.)

## Thanks to
sjchoi86 https://github.com/sjchoi86

# Main Codes

## char_rnn_train.py
note for training using korean bible corpus.  
data 파일을 읽어서 트레이닝 처리
기본 데이터로 저작권이 없는 성경말뭉치를 사용합니다.

## char_rnn_test_sj.ipynb
notebook for test sampling   
테스트 셈플링 코드 

## model.py
RNN network manifulation code  

## data_loader.py 
Data loading, parsing, batch slicing class  

# Legacy Codes(not using)
## conv2utf8.py
수집한 텍스트 파일 인코딩이 제각각인 관계로 utf8로 통합 변환하는코드.
10KB 로딩 후 인코딩 추측, 컨버팅 해서 저장 하는 코드
data/origin/*.txt를 모두 처리 후 data/conv/*.txt로 저장
인코딩 추측을 위해서 chardet 라이브러리에 의존
인코딩 추측을 실패한 경우 EUC_KR로 처리 후 확장자에 ".unknown" 추가
추측 확률이 0.8 이하일 경우 확장자에 ".notsure"추가 

# 출력 셈플 (96000 iter)

## 셈플 1
-- RESULT --
세상은 그 도모와 종비가의 행한 능력을 벞이리니 다볼라가 루스보다 의의 창상에서 취하지 말 것임이니라
그 생이 딸에 하기를 너희가 계명을 적어 그 형제 중으라 새끼를 수고에 굴라 나를 두는 두고 몸의 길을 벗겨다 하니 내가 번제를 행하느냐 하느냐
요나단이 만을 지켜 교회를 주되 서서 무슨 백성의 시을 미치고 배답하여 잔가도 그 위를 받는 것이 어금에 흠대를 멸저 온지라
저가 땅을 받고 일례를 당하였사오니 너희가 깨끗한 자에게 무두의 행계를 얻게 서편에서 목반 아비로 가서 그를 불설함이라
여자가 반드시 가질 때에라 그대 아비는 무릎은 하나님 여호와께 듣이든져 과하신 기이한 죄리와 유리하신 곳이 됨이 처장하니라
재나무들은 무할지라도 유다를 좇지라 애굽에서 나타

## 셈플 2
-- RESULT --
세상은 놀 열나도 위에서 불을 지워 드러내리라
대하지 않기를 뉘게 있도록 너희가 속죄하랴
보라 그 곡식 경계가 아니요 그 열국과 왕으로 그와 함께 욕금보 도지하는 전에는 마시리 그가 벧의 라한 것을 책망케 하느니라
사람아 애굽의 부이나는 달랍히 혈례줄을 가져다감고 사람이 나와서 죽이더라
때로 네 앞에서 내려올 것이 된 자여 허다지던 온 것이니라
너희는 블레셋 사람의 믿주도 보지 못하리라 하여 나의 전면에 완방과 부정씩 받으시리라
내 이 땅 백성은 의운 두 범죄하리라
너희가 삼수의 구전한 올라가라 하고자 그러하는 집을 섞었더라
고비파나귀편이로 바다 모으면 아니니 이는 하나가 이루기를 저주하고 공궤하시겠노라
모두 찌끌도 기도치 아니하더라
또 지금을 되지 않고 이 
