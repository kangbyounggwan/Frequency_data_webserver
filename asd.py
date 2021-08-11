"""
    RobotServoGun-ML-Service 
    ~~~~~~~~~~~~~~~~~~~~~~~
    Robot Servo Gun Machine Learning Service 
    :copyright: 2019 Tachyontech Co. Ltd.
"""
from apscheduler.schedulers.background import BackgroundScheduler
import os
import io
import time
import logging
import json
from datetime import datetime
from base64 import b64encode
from flask import jsonify, request

import joblib
import gc
from models import Result, result_schema_, Robot, robot_schema_, Schedata, sche_schema_, Sensor, sensor_schema_, Train, \
    train_schema_, Count, count_schema_, Ad_count
from models import ma, db, app
from logging.config import dictConfig
from datetime import datetime
import anomal_model

import time
import pytz
import logging
import datetime
import itertools
from urllib.parse import urlparse
import pickle
import pymysql
from flask import Flask, render_template, render_template, redirect, request, url_for
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms import PasswordField, SelectField
from wtforms.fields.html5 import DateField
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms.validators import DataRequired, EqualTo
from flask import session
name ='41:08:a1:0b'
temper_ = Schedata.query.filter_by(sensor_sensor_name=name).order_by(
    Schedata.data_id.desc()).first()
if temper_ == None:
    t = 0

    print(t)
else:
    print(temper_.temperate)