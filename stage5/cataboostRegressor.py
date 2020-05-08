from datetime import datetime

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class CataboostRegressor(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y


def computeData(X, y):
    # dividing data set into training 70% data set and 30% test data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Cataboost Regression :")
    start_time = datetime.now()
    print("For the training data Set:\n")
    catBoost = CatBoostRegressor(iterations=10000, loss_function='RMSE', boosting_type='Ordered')
    catBoost.fit(X_train, y_train, silent=True)
    y_train_pred = catBoost.predict(X_train)
    print('MSE train : %.3f\n' % (mean_squared_error(y_train, y_train_pred)))
    print('R^2 train: %.3f\n ' % (r2_score(y_train, y_train_pred)))
    print(catBoost.best_score_)
    print("running time is " + str(datetime.now() - start_time))
    start_time = datetime.now()
    print("For the testing data Set:\n")
    catBoost = CatBoostRegressor(iterations=10000, loss_function='RMSE', boosting_type='Ordered')
    catBoost.fit(X_test, y_test, silent=True)
    y_test_pred = catBoost.predict(X_test)
    print('MSE test : %.3f\n' % (mean_squared_error(y_test, y_test_pred)))
    print('R^2 test: %.3f\n ' % (r2_score(y_test, y_test_pred)))
    print(catBoost.best_score_)
    print("running time is " + str(datetime.now() - start_time))



