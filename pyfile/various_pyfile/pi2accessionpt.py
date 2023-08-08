import pandas as pd
import os
import shutil

# CSV 파일 읽기
df = pd.read_csv('/home/minkyoon/crohn/csv/original_csv/serial_accession_match.csv')

# 매핑 딕셔너리 생성
mapping = df.set_index('serial_number')['accession_number'].to_dict()

# pt 파일의 디렉토리 지정
pt_files_dir = '/home/minkyoon/first/testclam/feature/unifeature/pt_files'

# 출력 파일의 디렉토리 지정
output_dir = '/home/minkyoon/first/testclam/feature/unifeature/remission_accpt'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# pt 파일들에 대해
for file in os.listdir(pt_files_dir):
    # 파일 이름에서 serial_number 추출
    serial_number = int(os.path.splitext(file)[0])
    if serial_number in mapping:
        # 새로운 파일 이름 생성
        new_file_name = f'{mapping[serial_number]}.pt'
        # 새로운 파일 경로
        new_file_path = os.path.join(output_dir, new_file_name)
        # 원본 파일 경로
        old_file_path = os.path.join(pt_files_dir, file)
        # 파일 복사
        shutil.copy(old_file_path, new_file_path)
