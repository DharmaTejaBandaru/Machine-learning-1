from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegressor(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y


def computeData(X, y):
    # data scaling
    y= y.values.reshape(-1, 1)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    sc_x.fit(X)
    sc_y.fit(y)
    x_std = sc_x.transform(X)
    y_std = sc_y.transform(y).flatten()

    # dividing data set into training 70% data set and 30% test data set
    X_train, X_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.3, random_state=42)

    start_time = datetime.now()
    print("Linear Regression :")
    print("For the training data Set:\n")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    print('Slope : %.3f ' % model.coef_[0])
    print('Intercept : %.3f' % model.intercept_)
    print('MSE train : %.3f\n' % (mean_squared_error(y_train, y_train_pred)))
    print('R^2 train: %.3f\n ' % (r2_score(y_train, y_train_pred)))
    print("running time is " + str(datetime.now() - start_time))
    start_time = datetime.now()
    print("For the testing data Set:\n")
    model1 = LinearRegression()
    model1.fit(X_test, y_test)
    y_test_pred = model1.predict(X_test)
    print('Slope : %.3f ' % model1.coef_[0])
    print('Intercept : %.3f' % model.intercept_)
    print('MSE test : %.3f\n' % (mean_squared_error(y_test, y_test_pred)))
    print('R^2 test: %.3f\n ' % (r2_score(y_test, y_test_pred)))
    print("running time is " + str(datetime.now() - start_time))



