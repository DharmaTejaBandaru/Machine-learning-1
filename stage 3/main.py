# data pre-processing
import pandas as pd

# math operations
import numpy as np

# data scaling
from sklearn.preprocessing import StandardScaler

# importing zip file
import zipfile


# A function to generate some statistical data
def gen_features(X):
    strain = [X.mean(), X.std(), X.min(), X.max(), X.kurtosis(), X.skew(), np.quantile(X, 0.01)]
    return pd.Series(strain)


zf = zipfile.ZipFile('LANL-Earthquake-Prediction.zip')
train = pd.read_csv(zf.open('train.csv'), iterator=True, chunksize=150_000, nrows=600000,
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
X_train = pd.DataFrame()
Y_train = pd.Series()
for df in train:
    ch = gen_features(df['acoustic_data'])
    X_train = X_train.append(ch, ignore_index=True)
    Y_train = Y_train.append(pd.Series(df['time_to_failure'].values[-1]))

scalar = StandardScaler()
scalar.fit(X_train)
X_train_scaled = scalar.transform(X_train)
print(X_train_scaled)

# TODO choosing models