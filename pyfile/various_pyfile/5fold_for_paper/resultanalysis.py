import pandas as pd

df=pd.read_csv('/home/minkyoon/crohn/normal_resnet/hardvoting7,1,2/voting_results_5fold.csv')

df[df['voting_type']=='hard']['accuracy'].mean()