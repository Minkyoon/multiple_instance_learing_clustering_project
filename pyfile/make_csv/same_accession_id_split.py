import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('/home/minkyoon/crohn/csv/label_data/hb/hb_label_data_clear_under4.csv')



unique_accession_numbers = data['accession_number'].unique()

train_acc_nums, rest_acc_nums = train_test_split(unique_accession_numbers, test_size=0.2, random_state=42)
valid_acc_nums, test_acc_nums = train_test_split(rest_acc_nums, test_size=0.5, random_state=42)

train_data = data[data['accession_number'].isin(train_acc_nums)]
valid_data = data[data['accession_number'].isin(valid_acc_nums)]
test_data = data[data['accession_number'].isin(test_acc_nums)]





train_data.to_csv('train.csv', index=False)
valid_data.to_csv('valid.csv', index=False)
test_data.to_csv('test.csv', index=False)

test_data['accession_number'].unique()