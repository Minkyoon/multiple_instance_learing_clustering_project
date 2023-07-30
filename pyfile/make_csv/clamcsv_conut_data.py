import pandas as pd

# CSV 파일 위치
file_path = "/home/minkyoon/first/CLAM/dataset_csv/new_output.csv"

# CSV 파일을 pandas 데이터프레임으로 읽기
df = pd.read_csv(file_path)

# 'label' 컬럼 별 개수 확인
label_counts = df['label'].value_counts()

# 결과 출력
print(label_counts)