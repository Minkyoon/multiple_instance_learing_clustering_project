import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/clam/hb/original/hb_label_data.csv')

# "accession_number" 열에서 각 값의 개수 계산
count = df['accession_number'].value_counts()

# 결과 출력
print(count)


count[-84:]
len(count)


### 4개 이하인행 지우기


import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('/home/minkyoon/crohn/pyfile/make_csv/hb_label_data.csv')

# "accession_number" 열에서 각 값의 개수 계산
counts = df['accession_number'].value_counts()

# 데이터 개수가 4개 이하인 'accession_number' 찾기
to_remove = counts[counts <= 4].index

# 데이터 개수가 4개 이하인 행 지우기
df = df[~df['accession_number'].isin(to_remove)]


df.to_csv('hb_label_data_clear_under4.csv', index=False)
# 결과 확인
print(df)




import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/clam/hb/original/hb_label_data.csv')

# "accession_number" 열에서 각 값의 개수 계산
counts = df['accession_number'].value_counts()

# 데이터 개수가 4개 이하인 'accession_number' 찾기
to_remove = counts[counts <= 11].index

# 데이터 개수가 4개 이하인 행 지우기
df = df[~df['accession_number'].isin(to_remove)]


df.to_csv('hb_label_data_clear_under11.csv', index=False)
# 결과 확인
print(df)


## tco2
import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/clam/tco2/original/tCO2_label_data.csv')

# "accession_number" 열에서 각 값의 개수 계산
counts = df['accession_number'].value_counts()

# 데이터 개수가 4개 이하인 'accession_number' 찾기
to_remove = counts[counts <= 11].index

# 데이터 개수가 4개 이하인 행 지우기
df = df[~df['accession_number'].isin(to_remove)]


df.to_csv('tco2_label_data_clear_under11.csv', index=False)
# 결과 확인
print(df)
