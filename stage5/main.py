# math operations
import zipfile
# for plots

# math operations
import numpy as np
import pandas as pd
import sys

# collecting classifier name from command line argument
classifier = sys.argv[1]




# A function to generate some statistical data
def gen_features(X):
    strain = [X.mean(), X.std(), X.min(), X.max(), X.kurtosis(), X.skew(), np.quantile(X, 0.01)]
    return pd.Series(strain)


try:
    zf = zipfile.ZipFile('LANL-Earthquake-Prediction.zip')
    train = pd.read_csv(zf.open('train.csv'), iterator=True, nrows=6000000, chunksize=150_000,
                        dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
except FileNotFoundError as e:
    print("Entered invalid file name")
    sys.exit()

X = pd.DataFrame()
y = pd.Series()
for df in train:
    ch = gen_features(df['acoustic_data'])
    X = X.append(ch, ignore_index=True)
    y = y.append(pd.Series(df['time_to_failure'].values[-1]))

# print(X.shape)
# sys.exit()


# selecting classifier based on command line argument
if classifier == "catboost":
    import cataboostRegressor

    cataboostRegressor.computeData(X, y)
elif classifier == "dtree":
    import decisionTreeRegressor

    decisionTreeRegressor.computeData(X, y)
elif classifier == "linearR":
    import linearRegressor

    linearRegressor.computeData(X, y)
elif classifier == "rforest":
    import randomForestRegressor

    randomForestRegressor.computeData(X, y)
elif classifier == "svm":
    import supportVectorRegressor

    supportVectorRegressor.computeData(X, y)
else:
    print("invalid model entry")
