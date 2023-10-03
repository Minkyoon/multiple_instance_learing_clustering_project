import pandas as pd

# 첫 번째 파일에서 slide id 읽기
file1_path = "/home/minkyoon/CLAM2/data/processed/remission_under_10.csv"
df1 = pd.read_csv(file1_path)
slide_ids_1 = set(df1['slide_id'].tolist())

# 두 번째 파일에서 slide id 읽기
file2_path = "/home/minkyoon/CLAM2/splits/remission_multimodal_stratified_721/splits_0.csv"
df2 = pd.read_csv(file2_path)

# 'train', 'val', 'test' 열에서 slide id 값들을 가져와서 하나의 set으로 합치기
slide_ids_2 = set(df2['train'].tolist() + df2['val'].tolist() + df2['test'].tolist())

# 두 번째 파일에만 있는 slide id 찾기
unique_slide_ids_2 = slide_ids_2 - slide_ids_1

print(unique_slide_ids_2)
