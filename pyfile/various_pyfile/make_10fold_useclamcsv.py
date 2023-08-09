import pandas as pd
import numpy as np
import os

# Define the directory
dir_path = "/home/minkyoon/crohn/csv/clam/remission/clam_startified5fold"

# Iterate through the 5 csv files
for i in range(5):
    # Load each split file
    df_split = pd.read_csv(os.path.join(dir_path, f"splits_{i}.csv"))

    # Load the original dataset
    df_original = pd.read_csv('/home/minkyoon/crohn/csv/label_data/remission/remission_label.csv')

    # Convert to int
    df_original['accession_number'] = df_original['accession_number'].astype(int)

    # Create empty DataFrame for each split
    df_train = pd.DataFrame(columns=df_original.columns)
    df_val = pd.DataFrame(columns=df_original.columns)
    df_test = pd.DataFrame(columns=df_original.columns)

    # Populate the DataFrame
    for idx, row in df_split.iterrows():
        # Convert row to string and then split
        if pd.notna(row['train']):
            train_nums = list(map(int, map(float, str(row['train']).split(','))))
            df_train = pd.concat([df_train, df_original[df_original['accession_number'].isin(train_nums)]])
        if pd.notna(row['val']):
            val_nums = list(map(int, map(float, str(row['val']).split(','))))
            df_val = pd.concat([df_val, df_original[df_original['accession_number'].isin(val_nums)]])
        if pd.notna(row['test']):
            test_nums = list(map(int, map(float, str(row['test']).split(','))))
            df_test = pd.concat([df_test, df_original[df_original['accession_number'].isin(test_nums)]])

    # Save the new split csv files
    output_dir = "/home/minkyoon/crohn/csv/normal_resnet/5fold_resnet_for_startified5fold"
    df_train.to_csv(os.path.join(output_dir, f"train_fold_{i}.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, f"val_fold_{i}.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, f"test_fold_{i}.csv"), index=False)
