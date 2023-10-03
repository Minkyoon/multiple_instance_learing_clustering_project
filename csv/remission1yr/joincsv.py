import pandas as pd

# 첫 번째 CSV 파일 읽기
df1 = pd.read_csv("/home/minkyoon/crohn/csv/remission1yr/accesion_lab_PCDAI_serial_20230628_1.csv")

# 두 번째 CSV 파일 읽기
df2 = pd.read_csv("/home/minkyoon/crohn/csv/remission1yr/remission_multiple_label_1yr_remission.csv")

# 'ID' 열을 기준으로 두 데이터프레임 병합
merged_df = pd.merge(df1, df2, on='ID', how='inner')  # 'inner'는 두 데이터프레임에 모두 존재하는 'ID'만 병합됩니다.

# 병합된 데이터프레임을 새 CSV 파일로 저장
merged_df.to_csv("/home/minkyoon/crohn/csv/remission1yr/merged.csv", index=False)


## merged를 다시 join
df3 = pd.read_csv("/home/minkyoon/crohn/csv/remission1yr/merged.csv") 



df3_select = df3[['serial_number','label']]
df3_select = df3_select.rename(columns={'serial_number':'slide_id'})
df3_select['slide_id'] = df3_select['slide_id'].astype(int)



df3_select['label'].value_counts()


df4=pd.read_csv('/home/minkyoon/CLAM2/data/processed/remission_under_10.csv')



# 두 데이터프레임을 병합 (left join)
merged_df = pd.merge(df4, df3_select, how='left', on='slide_id')


merged_df.to_csv("/home/minkyoon/crohn/csv/remission1yr/merged_for_run.csv", index=False)


# 여기에서 실제쓸 merged


df_run=pd.read_csv('/home/minkyoon/crohn/csv/remission1yr/merged_for_run.csv')

df_for_concat=pd.read_csv('/home/minkyoon/CLAM2/data/processed/label.csv')

df_run=df_run[['slide_id', 'label']]

df_for_concat=df_for_concat.rename(columns={'label':'3month_label'})



merged_df = pd.merge(df_for_concat, df_run,  how='left', on='slide_id')
merged_df = merged_df.drop(columns=['3month_label'])
merged_df.to_csv('label.csv')
