from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class DTRegressor(object):
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

    print("Decision tree Regression :")
    start_time = datetime.now()
    print("For the training data Set:\n")
    dtree = DecisionTreeRegressor(max_depth=3)
    dtree.fit(X_train, y_train)
    y_train_pred = dtree.predict(X_train)
    print('MSE train : %.3f\n' % (mean_squared_error(y_train, y_train_pred)))
    print('R^2 train: %.3f\n ' % (r2_score(y_train, y_train_pred)))
    print("running time is " + str(datetime.now() - start_time))
    start_time = datetime.now()
    print("For the testing data Set:\n")
    dtree1 = DecisionTreeRegressor(max_depth=3)
    dtree1.fit(X_test, y_test)
    y_test_pred = dtree1.predict(X_test)
    print('MSE test : %.3f\n' % (mean_squared_error(y_test, y_test_pred)))
    print('R^2 test: %.3f\n ' % (r2_score(y_test, y_test_pred)))

    print("running time is " + str(datetime.now() - start_time))

