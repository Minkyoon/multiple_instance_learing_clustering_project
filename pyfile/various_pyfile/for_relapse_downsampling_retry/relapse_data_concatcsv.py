import pandas as pd

# CSV 파일 로드
relapse_label_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label.csv')
serial_accession_relapse_df = pd.read_csv('/home/minkyoon/crohn/csv/label_data/relapse/serial_accession_relapse_merged_downsampled.csv')

# relapse_label.csv에서 serial_accession_relapse_merged_downsampled.csv serial_number 열에 있는 accession_number만 남깁니다.
relapse_label_df = relapse_label_df[relapse_label_df['accession_number'].isin(serial_accession_relapse_df['serial_number'])]

# relapse_label.csv를 다시 저장
relapse_label_df.to_csv('/home/minkyoon/crohn/csv/label_data/relapse/relapse_label_filtered.csv', index=False)

# relapse_label.csv와 serial_accession_relapse_merged_downsampled.csv의 accession_number, serial_number가 다른 것을 찾습니다.
different_in_relapse_label = set(relapse_label_df['accession_number']).difference(set(serial_accession_relapse_df['serial_number']))
different_in_serial_accession = set(serial_accession_relapse_df['serial_number']).difference(set(relapse_label_df['accession_number']))

print("relapse_label.csv에서는 있지만 serial_accession_relapse_merged_downsampled.csv에서는 없는 accession_number:")
print(different_in_relapse_label)

print("serial_accession_relapse_merged_downsampled.csv에서는 있지만 relapse_label.csv에서는 없는 serial_number:")
print(different_in_serial_accession)

# 라벨의 개수 비교
relapse_label_count = relapse_label_df['label'].value_counts()
serial_accession_relapse_count = serial_accession_relapse_df['relapse'].value_counts()

print("relapse_label.csv의 라벨 개수:")
print(relapse_label_count)

print("serial_accession_relapse_merged_downsampled.csv의 라벨 개수:")
print(serial_accession_relapse_count)
