from openpyxl import load_workbook
from pandas import DataFrame
from itertools import islice
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
import datetime
import sys
from functools import partial
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, auc
from xgboost import XGBClassifier
from sklearn.utils import class_weight
from xgboost import plot_importance
from utils import createDirectory, preprocess_before_modelling_with_val, save_model_result
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score
import sklearn.metrics
from bayes_opt import BayesianOptimization
from warnings import filterwarnings
filterwarnings('ignore')
# set parameter of this python file
file_name = os.path.basename(__file__)
save_time = datetime.datetime.now().strftime(“%Y%m%d_%H%M”)
target = 'remission_under30'
createDirectory('/home/jsy/2023_retro_time_adaptive/data/model_results/' + file_name)
# sys.stdout = open('/home/jsy/2023_retro_time_adaptive/data/model_results/' + file_name + '/' + save_time + '.txt', 'w')
model_name= file_name + '_' + target
roc_aggr = []
f1s_aggr = []
data_size_aggr = []
relapse_counts_aggr = []
#follow_up 2023_05_30까지 되어 있음
#window i
# 2023_05_30 - i mo = last follow up date
def XGB_cv(max_depth,learning_rate, n_estimators, gamma
           , subsample, X_train, X_val, y_train, y_val,
           colsample_bytree, silent=True, nthread=-1):
    model = XGBClassifier(max_depth=int(max_depth),
                              learning_rate=learning_rate,
                              n_estimators=int(n_estimators),
                              silent=silent,
                              nthread=nthread,
                              gamma = gamma,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50, eval_metric='logloss', sample_weight=classes_weights)
    predicted = model.predict(X_val)
    predict_proba = model.predict_proba(X_val)
    # score = roc_auc_score(y_val, predict_proba)
    score = f1_score(y_val, predicted)
    # RMSE = cross_val_score(model, scaled_X_train1, y_train1, scoring='accuracy', cv=1).mean()
    return score
# 주어진 범위 사이에서 적절한 값을 찾는다.
pbounds = {'max_depth': (6, 12),
          'learning_rate': (0.0001, 0.0005),
          'n_estimators': (5000, 8000),
          'gamma': (0, 0.02),
          'subsample': (0.8, 1.0),
          'colsample_bytree' :(0.8, 1.0)
          }
ith_month = 3
df = pd.read_csv('/home/jsy/2023_retro_time_adaptive/data/processed/df_dx_lab_20230710.csv')
specific_date = pd.to_datetime('2023-05-30') - pd.DateOffset(months=ith_month)
df['dx_date'] = pd.to_datetime(df['dx_date'])
df = df[df['dx_date'] <= specific_date]
specific_date = '2015-01-01'
df['dx_date'] = pd.to_datetime(df['dx_date'])
df = df[df['dx_date'] >= specific_date]
df_htwt = pd.read_csv('/home/jsy/20230102_ibd_retro/data/processed/age_htwt_series.csv')
df_htwt.rename(columns={'연구등록번호':'ID', '날짜':'date', '성별':'gender'}, inplace=True)
df_htwt = df_htwt[['ID', 'date', 'z_ht', 'z_wt']]
df_htwt['date'] = df_htwt['date'].str[:10]
# df_htwt['date'] = pd.to_datetime(df_htwt['date'])
df = pd.merge(df, df_htwt, on=['ID', 'date'], how='outer')
df.to_csv('temp2.csv', index=False)
df_label = pd.read_csv('/home/jsy/2023_retro_time_adaptive/data/processed/total_lab_pcdai_manually.csv')
df_label['dx_date'] = pd.to_datetime(df_label['dx_date'])
df_label_2 = df_label.copy()
df_label_3 = df_label.copy()
df_label['dx_date_3moafter'] = df_label['dx_date'] + pd.Timedelta(weeks=12)
df_label_3 = df_label.copy()
df_label = df_label[['ID', 'date', 'dx_date', 'PCDAI', 'ab_date']]
df_label['date'] = pd.to_datetime(df_label['date'])
df_label['dx_date'] = pd.to_datetime(df_label['dx_date'])
# # 결과를 저장할 빈 데이터프레임 생성
# result_df = pd.DataFrame()
# # ID별로 데이터를 묶은 뒤 처리
# for _, group_data in df_label.groupby('ID'):
#     dx_date = group_data['dx_date'].iloc[0]  # 진단일
#     start_date = dx_date + pd.DateOffset(weeks=42)  # 11개월 이후
#     end_date = dx_date + pd.DateOffset(weeks=104)  # 12개월 이후
#     # 해당 범위에 속하는 데이터만 추출
#     # filtered_data = group_data[(group_data['date'] >= start_date) & (group_data['date'] <= end_date)]
#     filtered_data = group_data[(group_data['date'] >= start_date)]
#     # 결과 데이터프레임에 추가
#     result_df = pd.concat([result_df, filtered_data])
# print(result_df)
# 결과를 저장할 빈 데이터프레임 생성
result_df = pd.DataFrame(columns=['ID', 'label'])
# 결과를 저장할 list 생성
result_list = []
# ID별로 데이터를 묶은 뒤 처리
for id_, group_data in df_label.groupby('ID'):
    dx_date = group_data['dx_date'].iloc[0]  # 진단일
    start_date = dx_date + pd.DateOffset(weeks=42)  # 42주 이후
    end_date = dx_date + pd.DateOffset(weeks=78)  # 78주 이후
    # 해당 범위에 속하는 데이터만 추출
    filtered_data = group_data[(group_data['date'] >= start_date) & (group_data['date'] <= end_date)]
    # PCDAI 점수의 최대값에 따라 레이블을 지정
    max_value = filtered_data['PCDAI'].max() if not filtered_data.empty else None
    label = 1 if max_value and max_value >= 10 else 0
    result_list.append({'ID': id_, 'label': label})
# list를 DataFrame으로 변환
result_df = pd.DataFrame(result_list)
result_df.to_csv('remission_multiple_label_1yr_remission.csv', index=False)