import pandas as pd
import numpy as np
from rrcforest import RobustRandomCutForest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import time
import joblib
import os
from models import Train
basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE_PATH = basedir + '/ModelFiles/'




def anomal_model_(item, detection_type, training_ratio,samples):
    data = []
    x = pd.Series(np.fromstring(item, dtype=float, sep=','))
    data.append(x)
    np.array(data)
    data = np.array(data)
    data = data.reshape(-1,samples)
    data= pd.DataFrame(data)
    if detection_type == 'novelty':
        X_normal = data
        cut = int(len(X_normal) * training_ratio)
        X_train = X_normal[:cut]
        X_test = data

        return X_train,X_test

    elif detection_type == 'both':
        cut = int(len(data) * training_ratio)
        X_normal = data.sample(frac=1)
        X_train = X_normal[:cut]
        X_test = data

        return X_train,X_test


def algorithm_train(X_train,X_test,select):
    if select == 'rrcf':
        algorithm = RobustRandomCutForest(n_estimators=200, n_jobs=-1, max_samples=10,contamination=0.05)
    elif select == 'lof':
        algorithm = LocalOutlierFactor(n_neighbors=100,contamination=0.05, novelty=True)
    else:
        algorithm = IsolationForest(n_estimators=200, n_jobs=-1, max_samples=10,contamination=0.05)
    algorithm.fit_predict(X_train)
    start_time = time.time()
    result = algorithm.predict(X_test)
    print(X_train.shape)
    anomaly_score = -algorithm.score_samples(X_test)
    print(result)

    return algorithm, anomaly_score, result


def algorithm_test(data,sample,model):
    path = MODEL_FILE_PATH + model
    X_train_ = Train.query.filter_by(model_name=model).first()
    data = np.array(data)
    x = pd.Series(np.fromstring(X_train_.data_set, dtype=float, sep=','))


    x_train = np.array(x)
    with open(path, "rb") as file:
         model = joblib.load(file)
    raw = []
    data_ = data[-1]


    for j in range(len(data_)):
        raw.append(data_[j])
    data = np.array(raw)
    data = data.reshape(-1,sample)
    x_train = x_train.reshape(-1,sample)
    data = pd.DataFrame(data)
    result = model.predict(data)
    score = - model.score_samples(data)
    print(score)
    print(result)
    return result,score