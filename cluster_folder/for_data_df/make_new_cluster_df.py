import pandas as pd

# csv 파일 읽기
df1 = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_clustering/class3_clustering.csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_clustering/class1_clustering.csv')

# 조건에 맞는 행 선택
df1_selected = df1[df1['label'] == 1.0]
df2_selected = df2[df2['label'] == 0.0]

# 두 DataFrame 합치기
df_united = pd.concat([df1_selected, df2_selected])

# 새로운 csv 파일로 저장 csv 파일이름은 앞쪽에 있는것을 relapse 라벨 추출한것으로 하자!
df_united.to_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_united/class3_class1_clustering.csv', index=False)



import pandas as pd

# 기존 CSV 파일 읽기
df1 = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_clustering/class1_clustering.csv')
df2 = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_clustering/class2_clustering.csv')

# 새로운 CSV 파일 읽기
df3 = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_clustering/class3_clustering.csv')
df4 = pd.read_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_clustering/class4_clustering.csv')

# 조건에 맞는 행 선택
df1_selected = df1[df1['label'] == 1.0]
df2_selected = df2[df2['label'] == 0.0]
df3_selected = df3[df3['label'] == 1.0]
df4_selected = df4[df4['label'] == 0.0]

# 모든 DataFrame 합치기
df_united = pd.concat([df1_selected, df2_selected, df3_selected, df4_selected])
#class13_no_ 이면 relapse 라벨 끼리 분류
# 새로운 CSV 파일로 저장
df_united.to_csv('/home/minkyoon/crohn/csv/clam/clustering/relapse_resnet/class_united/class13_no_clustering.csv', index=False)
