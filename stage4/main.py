from datetime import datetime
# data pre-processing
import pandas as pd

# math operations
import numpy as np
# for plots
import matplotlib.pyplot as plt

# data scaling
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing zip file
import zipfile

# plotting Acostic data and time to failure
from sklearn.tree import DecisionTreeRegressor


def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df,
                      title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='tab:orange')
    ax1.set_ylabel('acoustic data', color='tab:orange')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='tab:blue')
    ax2.set_ylabel('time to failure', color='tab:blue')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)


# A function to generate some statistical data
def gen_features(X):
    strain = [X.mean(), X.std(), X.min(), X.max(), np.quantile(X, 0.01)]
    return pd.Series(strain)


zf = zipfile.ZipFile('LANL-Earthquake-Prediction.zip')
train = pd.read_csv(zf.open('train.csv'), nrows=20000,
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

X = train['acoustic_data'].values.reshape(-1, 1)
y = train['time_to_failure'].values.reshape(-1, 1)
# plot_acc_ttf_data(X, y)  used for plotting data
sc_x = StandardScaler()
sc_y = StandardScaler()
sc_x.fit(X)
sc_y.fit(y)
x_std = sc_x.transform(X)
y_std = sc_y.transform(y).flatten()

X_train, X_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.3, random_state=0)
print("Linear Regression :")
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print('Linear Regression')
print('Slope : %.3f ' % model.coef_[0])
print('Intercept : %.3f' % model.intercept_)
print('MSE train:  %.3f\nMSE test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f\nR^2 test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))
print("Decision tree Regression :")
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)
y_train_pred = tree.predict(X_train)
print('Non linear Regression - Decision Tree Regressor')
print('MSE train : %.3f\nMSE test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f\nR^2  test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))

print("Random forest Regression :")
tree = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=10)
tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)
y_train_pred = tree.predict(X_train)
print('Non linear Regression - Random forest Regressor')
print('MSE train : %.3f\nMSE test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f\nR^2  test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))


