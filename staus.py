import os
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import time


def status():
    path_to_dir_train = 'C:\\Users\\ST200423\\Desktop\\'
    path_to_train = os.path.join(path_to_dir_train, 'data_.csv')
    data = pd.read_csv(path_to_train)
    data_ = data.sort_index(ascending=False)
    last_data = data_.iloc[0]
    time = last_data["time"]
    status = last_data["status"]
    step = last_data['step']
    speed = last_data["speed"]
    return time, status, speed, step
