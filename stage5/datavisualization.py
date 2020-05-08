!pip install kaggle

pip install numpy==1.18.3

pip install catboost

#data preprocessing
import pandas as pd
#math operations
import numpy as np
#machine learning
from catboost import CatBoostRegressor, Pool
#data scaling
from sklearn.preprocessing import StandardScaler
#hyperparameter optimization
from sklearn.model_selection import GridSearchCV
#support vector machine model
from sklearn.svm import NuSVR, SVR
#kernel ridge model
from sklearn.kernel_ridge import KernelRidge
#data visualization
import matplotlib.pyplot as plt

# Colab's file access feature
from google.colab import files

#retrieve uploaded file
uploaded = files.upload()
  
# Then move kaggle.json into the folder where the API expects to find it.
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

#download earthquake data, will take 30-60 seconds
!kaggle competitions download -c LANL-Earthquake-Prediction --force

#unzip training data for usage, will take about 5 minutes (its big)
!ls
!unzip train.csv.zip
!ls

#Extract training data into a dataframe for further manipulation
train = pd.read_csv('train.csv', nrows=60000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

#print first 10 entries
train.head(10)

fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of acoustic_data and time_to_failure. First 2% of data")
plt.plot(train['acoustic_data'].values[:12582910], color='orange')
ax1.set_ylabel('acoustic_data', color='orange')
plt.legend(['acoustic_data'])
ax2 = ax1.twinx()
plt.plot(train['time_to_failure'].values[:12582910], color='blue')
ax2.set_ylabel('time_to_failure', color='blue')
plt.legend(['time_to_failure'], loc=(0.875, 0.9))
plt.grid(False)

from scipy import stats
import seaborn as sns
from scipy.stats import norm
def single_timeseries(final_idx, init_idx=0, step=1, title="",
                      color1='orange', color2='blue'):
    idx = [i for i in range(init_idx, final_idx, step)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('index')
    ax1.set_ylabel('Acoustic data')
    ax2.set_ylabel('Time to failure')
    p1 = sns.lineplot(data=train.iloc[idx].acoustic_data.values, ax=ax1, color=color1)
    p2 = sns.lineplot(data=train.iloc[idx].time_to_failure.values, ax=ax2, color=color2)
single_timeseries(10000, title="Ten thousand rows")

def single_timeseries(final_idx, init_idx=0, step=1, title="",
                      color1='orange', color2='blue'):
    idx = [i for i in range(init_idx, final_idx, step)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('index')
    ax1.set_ylabel('Acoustic data')
    ax2.set_ylabel('Time to failure')
    p1 = sns.lineplot(data=train.iloc[idx].acoustic_data.values, ax=ax1, color=color1)
    p2 = sns.lineplot(data=train.iloc[idx].time_to_failure.values, ax=ax2, color=color2)
single_timeseries(60000000, step=1000, title="All training data")

figure, axes1 = plt.subplots(figsize=(18,10))

plt.title("Seismic Data Trends with 5% sample of original data")

plt.plot(train['acoustic_data'], color='orange')
axes1.set_ylabel('Acoustic Data', color='orange')
plt.legend(['Acoustic Data'])

axes2 = axes1.twinx()
plt.plot(train['time_to_failure'], color='b')
axes2.set_ylabel('Time to Failure', color='b')
plt.legend(['Time to Failure'])

