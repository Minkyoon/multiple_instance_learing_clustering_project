import pandas as pd
import numpy as np

filepaths = ['/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normal_remission_for_multimodaal_100/splits_0.csv',
             '/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normal_remission_for_multimodaal_100/splits_1.csv',
             '/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normal_remission_for_multimodaal_100/splits_2.csv',
             '/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normal_remission_for_multimodaal_100/splits_3.csv',
             '/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normal_remission_for_multimodaal_100/splits_4.csv',
            
             ]

for filepath in filepaths:
    # csv 파일 로드
    df = pd.read_csv(filepath)

    # 변경된 csv 파일 저장 (기존 파일 덮어쓰기)
    df.to_csv(filepath, index=False, float_format='%.0f')
