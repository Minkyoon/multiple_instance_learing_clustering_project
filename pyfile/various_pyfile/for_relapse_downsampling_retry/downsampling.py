import pandas as pd
import numpy as np
import pandas as pd
import numpy as np


# 데이터 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_with_age_sex.csv')

# 레이블 0과 1의 데이터를 분리
df_0 = df[df['relapse'] == 0.0]
df_1 = df[df['relapse'] == 1.0]

# 각 나이대 별로 그룹화
age_groups = df['age'].unique()

# 결과를 저장할 데이터프레임 초기화
df_result = pd.DataFrame()

# 각 나이 그룹별로 처리
for age in age_groups:
    for sex in ['M', 'W']:
        df_0_same_age_sex = df_0[(df_0['age'] == age) & (df_0['sex'] == sex)]
        df_1_same_age_sex = df_1[(df_1['age'] == age) & (df_1['sex'] == sex)]

        # 레이블 1의 개수와 레이블 0의 개수를 동일하게 맞춤
        if len(df_0_same_age_sex) > len(df_1_same_age_sex):
            df_0_same_age_sex = df_0_same_age_sex.sample(len(df_1_same_age_sex))
        else:
            df_1_same_age_sex = df_1_same_age_sex.sample(len(df_0_same_age_sex))
        
        df_same_age_sex = pd.concat([df_0_same_age_sex, df_1_same_age_sex])
        
        df_result = pd.concat([df_result, df_same_age_sex])
        
        
        
        
        
        
import pandas as pd
import numpy as np

# 데이터 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_with_age_sex.csv')

# 레이블 0과 1의 데이터를 분리
df_0 = df[df['relapse'] == 0.0]
df_1 = df[df['relapse'] == 1.0]

# 결과를 저장할 데이터프레임 초기화
df_result = pd.DataFrame()

# 레이블 1의 데이터를 기준으로 레이블 0의 데이터 매칭
for i, row in df_1.iterrows():
    df_0_same_sex = df_0[df_0['sex'] == row['sex']]
    df_0_similar_age = df_0_same_sex[(df_0_same_sex['age'] >= row['age'] - 1) & (df_0_same_sex['age'] <= row['age'] + 1)]
    
    # 레이블 1의 개수와 레이블 0의 개수를 동일하게 맞춤
    if len(df_0_similar_age) > 1:
        df_0_similar_age = df_0_similar_age.sample(1)
    
    df_result = pd.concat([df_result, df_0_similar_age, df_1.loc[[i]]])

# 결과 저장
df_result.to_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_downsampled2.csv', index=False)

df1=pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_downsampled.csv')








import pandas as pd
import numpy as np

# 데이터 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_with_age_sex.csv')

# 레이블 0과 1의 데이터를 분리
df_0 = df[df['relapse'] == 0.0]
df_1 = df[df['relapse'] == 1.0]

# 결과를 저장할 데이터프레임 초기화
df_result = pd.DataFrame()

# 레이블 1의 데이터를 기준으로 레이블 0의 데이터 매칭
for i, row in df_1.iterrows():
    df_0_same_sex = df_0[df_0['gender'] == row['gender']]
    df_0_similar_age = df_0_same_sex[(df_0_same_sex['age'] >= row['age'] - 1) & (df_0_same_sex['age'] <= row['age'] + 1)]
    
    # 레이블 1의 개수와 레이블 0의 개수를 동일하게 맞춤
    if len(df_0_similar_age) > 1:
        df_0_similar_age = df_0_similar_age.sample(1)
    
    df_result = pd.concat([df_result, df_0_similar_age, df_1.loc[[i]]])

# 만약 레이블 1의 개수가 레이블 0의 개수보다 많다면, 부족한 만큼 레이블 0에서 무작위로 추가로 선택
if len(df_1) > len(df_result[df_result['relapse'] == 0.0]):
    df_0_remaining = df_0.drop(df_result[df_result['relapse'] == 0.0].index)
    df_0_additional = df_0_remaining.sample(len(df_1) - len(df_result[df_result['relapse'] == 0.0]), random_state=1)
    df_result = pd.concat([df_result, df_0_additional])

# 결과 저장
df_result.to_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_downsampled.csv', index=False)



## 최종적으로 라벨0, 1 동일하게 하도록하는것임 보통 이런건 맨밑에 부분에 정답이 있고 생각할것!


import pandas as pd
import numpy as np

# 데이터 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_with_age_sex.csv')

# 레이블 0과 1의 데이터를 분리
df_0 = df[df['relapse'] == 0.0]
df_1 = df[df['relapse'] == 1.0]

# 결과를 저장할 데이터프레임 초기화
df_result = pd.DataFrame()

# 레이블 1의 데이터를 기준으로 레이블 0의 데이터 매칭
for i, row in df_1.iterrows():
    df_0_same_sex = df_0[df_0['gender'] == row['gender']]
    df_0_similar_age = df_0_same_sex[(df_0_same_sex['age'] >= row['age'] - 1) & (df_0_same_sex['age'] <= row['age'] + 1)]
    
    # 만약 매칭되는 레이블 0의 행이 없다면 레이블 0에서 무작위로 선택
    if len(df_0_similar_age) == 0:
        df_0_similar_age = df_0_same_sex.sample(1)
    
    df_result = pd.concat([df_result, df_0_similar_age, df_1.loc[[i]]])

# 결과 저장
df_result.to_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_downsampled.csv', index=False)



import pandas as pd
import numpy as np

# 데이터 불러오기
df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_with_age_sex.csv')

# 레이블 0과 1의 데이터를 분리
df_0 = df[df['relapse'] == 0.0]
df_1 = df[df['relapse'] == 1.0]

# 결과를 저장할 데이터프레임 초기화
df_result = pd.DataFrame()

# 레이블 1의 데이터를 기준으로 레이블 0의 데이터 매칭
for i, row in df_1.iterrows():
    df_0_same_sex = df_0[df_0['gender'] == row['gender']]
    df_0_similar_age = df_0_same_sex[(df_0_same_sex['age'] >= row['age'] - 1) & (df_0_same_sex['age'] <= row['age'] + 1)]
    
    # 레이블 1의 개수와 레이블 0의 개수를 동일하게 맞춤
    if len(df_0_similar_age) > 0:
        df_0_similar_age = df_0_similar_age.sample(1)
        df_0 = df_0.drop(df_0_similar_age.index)
    else:
        df_0_similar_age = df_0.sample(1)
        df_0 = df_0.drop(df_0_similar_age.index)
    
    df_result = pd.concat([df_result, df_0_similar_age, df_1.loc[[i]]])

# 결과 저장
df_result.to_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_downsampled.csv', index=False)
