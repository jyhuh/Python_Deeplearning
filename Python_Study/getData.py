import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Danse, Activation
import datetime
#주식 데이터를 받기 위한 코드. 1일 1회 실행 예정

code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]

# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
code_df.종목코드 = code_df.종목코드.map('{:06d}'.format)
# 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.

code_df = code_df[['회사명', '종목코드']]
# 한글로된 컬럼명을 영어로 바꿔준다.

code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})

print(code_df.head())

def get_url(item_name, code_df):
    code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    url = 'http://finance.naver.com/item/sise_day.nhn?code=' + code.strip()
    print("요청 URL = {}".format(url))
    return url

# 일자데이터 url 가져오기
item_name = '쎄트렉아이'
url = get_url(item_name, code_df)
# 일자 데이터를 담을 df 라는 DataFrame 정의
df = pd.DataFrame()
# 1페이지에서 20페이지의 데이터만 가져오기

for page in range(1, 21):
    pg_url = '{url}&page={page}'.format(url=url, page=page)
    df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)

# df.dropna()를 이용해 결측값 있는 행 제거
df = df.dropna()

# 한글로 된 컬럼명을 영어로 바꿔줌
df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})

# 데이터의 타입을 int형으로 바꿔줌
df[['close', 'diff', 'open', 'high', 'low', 'volume']] \
    = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)

# 컬럼명 'date'의 타입을 date로 바꿔줌
df['date'] = pd.to_datetime(df['date'])
# 일자(date)를 기준으로 오름차순 정렬
df = df.sort_values(by=['date'], ascending=True)
# 상위 5개 데이터 확인
print(df.head())

'''
mid_prices = (df['high'].values + df['high'].values)/2

seq_len = 20
sequence_length = seq_len + 1
result = []

for index in range(len(mid_prices)-sequence_length):
    result.append(mid_prices[index:index+sequence_length])

normalized_data = []
for window in result:
    normalized_window = [((float(p)/float(window[0]))-1)for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

row = int(round(result.shape[0]*0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, 1]

print(x_train.shape)

print(x_test.shape)
'''
