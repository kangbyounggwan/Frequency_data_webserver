from datetime import datetime
from base64 import b64encode
from flask import Flask
from flask import jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
import pandas as pd
import numpy as np
import dill as pickle
import time
from models import Sensor, ma, db, app, Schedata, Train
import pymysql
import subprocess
import anomal_model
import json
import pytz
import datetime
import opc_ua
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from urllib.parse import urlparse
from opc_ua import conn


def opc_ua_key(sensor_name, sensor_axis):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("opcua").setLevel(logging.WARNING)

    # replace xx.xx.xx.xx with the IP address of your server
    serverIP = "25.27.135.161"
    serverUrl = urlparse('opc.tcp://{:s}:4840'.format(serverIP))

    # replace xx:xx:xx:xx with your sensors macId
    macId = sensor_name
    timeZone = "Asia/Seoul"  # local time zone
    now = datetime.datetime.now()
    dt_s = now - datetime.timedelta(minutes=560)
    dt_e = now - datetime.timedelta(minutes=540)
    startTime = dt_s.strftime('%Y-%m-%d %H:%M:%S')
    TIME = dt_e.strftime('%Y-%m-%d %H:%M:%S')
    endTime = TIME
    starttime = pytz.utc.localize(
        datetime.datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S')
    )
    endtime = pytz.utc.localize(
        datetime.datetime.strptime(endTime, '%Y-%m-%d %H:%M:%S')
    )

    # acquire history data
    (values, dates) = opc_ua.DataAcquisition.get_sensor_data(
        serverUrl=serverUrl,
        macId=macId,
        browseName="vibration",
        starttime=starttime,
        endtime=endtime,
        axis_=sensor_axis
    )
    (temperatures, dates_) = opc_ua.DataAcquisition.get_sensor_data(
        serverUrl=serverUrl,
        macId=macId,
        browseName="boardTemperature",
        starttime=starttime,
        endtime=endtime,
        axis_=sensor_axis
    )
    return dates, values, timeZone, temperatures


def schadule_opc():
    sensor = Sensor.query.order_by(Sensor.sensor_name).all()
    sensor_name = []
    for i in range(len(sensor)):
        sensor_name.append(sensor[i].sensor_name)

    for name in sensor_name:
        print(name)
        # 마지막날짜
        curs = conn.cursor()
        curs.execute("SELECT date_stamp from sche_data where sensor_sensor_name = %s", name)
        date_raw = curs.fetchall()

        if len(date_raw) == 0:
            last_stamp = None
        else:
            data = date_raw[-1]
            date_raw_ = data[0]
            last_stamp = date_raw_

        print(last_stamp)

        sensor = Sensor.query.filter_by(sensor_name=name).first()
        samples = sensor.sample
        sensor_axis = sensor.axis
        mo = Train.query.filter_by(sensor_sensor_name=name).all()
        model = []
        num = []
        for i in mo:
            if i.model_name != None:
                model.append(i.model_name)
                num.append(i.num)

        dates, values, timeZone, temperatures = opc_ua_key(name, sensor_axis)
        if len(temperatures) != 0:
            print(temperatures[-1])
        data, len_ = opc_ua.DataAcquisition.plotly(dates, values)

        model = model[-1]
        num = num[-1]

        title = []
        for j in range(len(dates)):
            title_ = datetime.datetime.strptime(dates[j], '%Y-%m-%d %H:%M:%S')
            title_ = title_.replace(tzinfo=pytz.timezone('UTC')).astimezone(pytz.timezone(timeZone))
            title.append(title_.strftime('%Y-%m-%d %H:%M:%S'))

        if len(title) == 0:
            continue

        title_last = title[-1]

        if last_stamp != None:
            print(last_stamp)
            print(title_last)

        else:
            temperatures = temperatures[-1]
            print(temperatures)
            sample = samples
            result, score = anomal_model.algorithm_test(data, sample, model)
            opc_ua.DataAcquisition.sche(data, title_last, result, score, name, model, num, temperatures)
            print('업로드 완료')

            continue

        # 마지막날짜 마지막title 비교
        if str(last_stamp) != str(title_last):
            sample = samples
            print(model)
            temperatures = temperatures[-1]
            result, score = anomal_model.algorithm_test(data, sample, model)
            opc_ua.DataAcquisition.sche(data, title_last, result, score, name, model, num, temperatures)

            print('업로드 완료')

        else:
            print('새로운 데이터없음')
            continue


scheduler = BackgroundScheduler()
job = scheduler.add_job(schadule_opc, 'interval', minutes=2)
scheduler.start()

while True:
    time.sleep(1)
