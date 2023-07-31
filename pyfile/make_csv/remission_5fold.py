import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

# Load data
df = pd.read_csv('/home/minkyoon/crohn/csv/end_to_end/remission/remission_label.csv')

# Add a new column with only the filename part of the filepath
df['filename'] = df['filepath'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))

# Sort by 'accession_number' first and then by 'filename'
df = df.sort_values(['accession_number', 'filename'])

# Remove the 'filename' column as it's not needed anymore
df = df.drop('filename', axis=1)

# Implement 5-fold cross-validation on the whole dataset
gkf = GroupKFold(n_splits=5)

for i, (train_valid_index, test_index) in enumerate(gkf.split(df, groups=df['accession_number'])):
    # Get train_valid and test subsets for the current fold
    train_valid_subset = df.iloc[train_valid_index]
    test_subset = df.iloc[test_index]
    
    # Split train_valid subset into train and valid
    train_subset, valid_subset = train_test_split(train_valid_subset, test_size=1/8, stratify=train_valid_subset['label'], random_state=42)
    
    # Save each fold as separate CSV files
    train_subset.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/train_fold{i}.csv', index=False)
    valid_subset.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/valid_fold{i}.csv', index=False)
    test_subset.to_csv(f'/home/minkyoon/crohn/csv/end_to_end/remission/5FOLD/test_fold{i}.csv', index=False)


