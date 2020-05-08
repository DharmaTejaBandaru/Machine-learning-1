from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SupportVectorRegressor(object):
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

    print("Support Vector Regression :")
    start_time = datetime.now()
    print("For the training data Set:\n")
    parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                   'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]
    svr1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_squared_error')
    svr1.fit(X_train, y_train)
    y_train_pred = svr1.predict(X_train)
    print('MSE train : %.3f\n' % (mean_squared_error(y_train, y_train_pred)))
    print('R^2 train: %.3f\n ' % (r2_score(y_train, y_train_pred)))
    print("Best CV score: {:.4f}".format(svr1.best_score_))
    print(svr1.best_params_)
    print("running time is " + str(datetime.now() - start_time))
    start_time = datetime.now()
    print("For the testing data Set:\n")
    parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                   'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]
    svr = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_squared_error')
    svr.fit(X_test, y_test)
    y_test_pred = svr.predict(X_test)
    print('MSE test : %.3f\n' % (mean_squared_error(y_test, y_test_pred)))
    print('R^2 test: %.3f\n ' % (r2_score(y_test, y_test_pred)))
    print("Best CV score: {:.4f}".format(svr.best_score_))
    print(svr.best_params_)
    print("running time is " + str(datetime.now() - start_time))
