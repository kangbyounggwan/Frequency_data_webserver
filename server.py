"""
    RobotServoGun-ML-Service 
    ~~~~~~~~~~~~~~~~~~~~~~~
    Robot Servo Gun Machine Learning Service 
    :copyright: 2019 Tachyontech Co. Ltd.
"""

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
import opc_ua


# 센서등록
class New_sensorForm(FlaskForm):
    sensor_name = StringField('sensor_name', validators=[DataRequired()])
    axis = StringField('axis', validators=[DataRequired()])
    locate = StringField('locate', validators=[DataRequired()])


# 로봇등록
class New_robotForm(FlaskForm):
    robot_name = StringField('robot_name', validators=[DataRequired()])
    robot_manufacture = StringField('robot_manufacture', validators=[DataRequired()])


# 회원가입
class RegisterForm(FlaskForm):
    user_id = StringField('user_id', validators=[DataRequired()])
    line_name = StringField('line_name', validators=[DataRequired()])
    user_name = StringField('user_name', validators=[DataRequired()])
    factory_name = StringField('user_name', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired(), EqualTo('password_2')])  # 비밀번호 확인
    password_2 = PasswordField('repassword', validators=[DataRequired()])


# ad회원가입
class ad_registerForm(FlaskForm):
    user_id = StringField('user_id', validators=[DataRequired()])
    user_name = StringField('user_name', validators=[DataRequired()])
    factory_name = StringField('user_name', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired(), EqualTo('password_2')])  # 비밀번호 확인
    password_2 = PasswordField('repassword', validators=[DataRequired()])


# 로그인
class LoginForm(FlaskForm):
    class UserPassword(object):
        def __init__(self, message=None):
            self.message = message

        def __call__(self, form, field):
            user_id = form['user_id'].data
            password = field.data
            count = Count.query.filter_by(user_id=user_id).first()
            session["status"] = 0
            if count == None:
                count = Ad_count.query.filter_by(user_id=user_id).first()
                session["status"] = 1
            if count.password != password:
                raise ValueError('Wrong Password!')

    user_id = StringField('user_id', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired(), UserPassword()])


# 날짜 선택
class ExampleForm(FlaskForm):
    dt = DateField('DatePicker', format='%Y-%m-%d')
    dta = DateField('DatePicker', format='%Y-%m-%d')


basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE_PATH = basedir + '/ModelFiles/'

detection_type = 'novelty'  # both
scaling = True
training_ratio = 1


# 로그인
@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if session['status'] == 0:
            print('{}가 로그인 했습니다.'.format(form.data.get('user_id')))
            user_id = form.data.get('user_id')
            session['user_id'] = user_id
            count = Count.query.filter_by(user_id=user_id).first()
            session['user_name'] = count.user_name
            session["line_name"] = count.line_name
            return redirect('/dashboard')
        else:
            print('{}가 로그인 했습니다.'.format(form.data.get('user_id')))
            user_id = form.data.get('user_id')
            session['user_id'] = user_id
            count = Ad_count.query.filter_by(user_id=user_id).first()
            session['user_name'] = count.user_name
            session['factory'] = count.factory_name
            return redirect('/all')
    return render_template('login.html', form=form)


# 로그아웃
@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return redirect('/')


# 대시보드
# GET(정보보기), POST(정보수정) 메서드 허용
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if session['status'] == 0:
        user_line = Count.query.filter_by(user_id=session['user_id']).first()

        line = user_line.line_name
        robot_card = []
        robot_id = []
        robot = Robot.query.filter_by(user_count_line_name=line).all()
        for robot_ in robot:
            robot_card.append(robot_.robot_name)
            robot_id.append(robot_.robot_id)

        # 테이블 정보
        r_name = []
        sensor_name = []
        sensor_axis = []
        sensor_locate = []
        sensor_id = []

        sensor_ = []
        for i in robot_id:
            sensor = Sensor.query.filter_by(robot_robot_id=i).all()
            sensor_.append(sensor)
            for j in range(len(sensor)):
                sensor_name.append(sensor[j].sensor_name)
                sensor_axis.append(sensor[j].axis)
                sensor_locate.append(sensor[j].locate)
                num = sensor[j].robot_robot_id
                sensor_id.append(sensor[j].sensor_id)
                rname = Robot.query.filter_by(robot_id=num).first()
                r_name.append(rname.robot_name)

        for i in range(len(sensor_)):
            data = sensor_[i]
            for j in range(len(data)):

                value = Result.query.filter_by(
                    sensor_sensor_name=data[j].sensor_name).order_by(Result.result_key.desc()).first()
                if value == None:
                    continue
                elif value.predict == -1:
                    sensor_[i] = -1
                    break
                else:
                    sensor_[i] = 0

        return render_template('index.html',
                               robot_card=robot_card,
                               robot_id=robot_id,
                               sensor_axis=sensor_axis,
                               sensor_name=sensor_name,
                               sensor_locate=sensor_locate,
                               sensor_result=sensor_,
                               sensor_id=sensor_id,
                               enumerate=enumerate,
                               r_name=r_name,
                               zip=zip,
                               line=line
                               )


# admin dashboard
@app.route('/dashboard_ad/<int:id>', methods=['GET', 'POST'])  # GET(정보보기), POST(정보수정) 메서드 허용
def dashboard_ad(id=None):
    robot = Robot.query.filter_by(robot_id=id).first()
    line = robot.user_count_line_name
    robot_card = []
    robot_id = []
    robot = Robot.query.filter_by(user_count_line_name=line).all()
    for robot_ in robot:
        robot_card.append(robot_.robot_name)
        robot_id.append(robot_.robot_id)
    # 테이블 정보
    r_name = []
    sensor_name = []
    sensor_axis = []
    sensor_locate = []
    sensor_id = []
    sensor_ = []
    for i in robot_id:
        sensor = Sensor.query.filter_by(robot_robot_id=i).all()
        sensor_.append(sensor)
        for j in range(len(sensor)):
            sensor_name.append(sensor[j].sensor_name)
            sensor_axis.append(sensor[j].axis)
            sensor_locate.append(sensor[j].locate)
            num = sensor[j].robot_robot_id
            sensor_id.append(sensor[j].sensor_id)
            rname = Robot.query.filter_by(robot_id=num).first()
            r_name.append(rname.robot_name)

    for i in range(len(sensor_)):
        data = sensor_[i]
        for j in range(len(data)):
            value = Result.query.filter_by(
                sensor_sensor_name=data[j].sensor_name).order_by(Result.result_key.desc()).first()
            if value == None:
                continue
            elif value.predict == -1:
                sensor_[i] = -1
                break
            else:
                sensor_[i] = 0
    print(sensor_)

    return render_template('index_admin.html',
                           robot_card=robot_card,
                           robot_id=robot_id,
                           sensor_axis=sensor_axis,
                           sensor_name=sensor_name,
                           sensor_locate=sensor_locate,
                           sensor_id=sensor_id,
                           enumerate=enumerate,
                           r_name=r_name,
                           id_=id,
                           zip=zip,
                           sensor_result=sensor_,
                           line=line
                           )


# 회원가입
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():  # 내용 채우지 않은 항목이 있는지까지 체크
        usertable = Count(user_id=None, line_name=None, password=None, user_name=None, approval_status=None,
                          factory_name=None)
        usertable.user_id = form.data.get('user_id').strip()
        usertable.line_name = form.data.get('line_name').strip()
        usertable.password = form.data.get('password').strip()
        usertable.user_name = form.data.get('user_name').strip()
        usertable.factory_name = form.data.get('factory_name').strip()
        usertable.approval_status = '0'

        db.session.add(usertable)  # DB저장
        db.session.commit()  # 변동사항 반영
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


# admin회원가입
@app.route('/ad_register', methods=['GET', 'POST'])
def ad_register():
    form = ad_registerForm()
    if form.validate_on_submit():  # 내용 채우지 않은 항목이 있는지까지 체크
        usertable = Ad_count(user_id=None, password=None, user_name=None, factory_name=None)
        usertable.user_id = form.data.get('user_id').strip()
        usertable.password = form.data.get('password').strip()
        usertable.user_name = form.data.get('user_name').strip()
        usertable.factory_name = form.data.get('factory_name').strip()

        db.session.add(usertable)  # DB저장
        db.session.commit()  # 변동사항 반영
        return redirect(url_for('login'))
    return render_template('ad_register.html', form=form)


# 로봇 등록 admin
@app.route('/dashboard/new_robot/<int:robot_id>', methods=['GET', 'POST'])
def new_robot_ad(robot_id=None):
    form = New_robotForm()
    line_ = Robot.query.filter_by(robot_id=id).first()
    line = line_.user_count_line_name
    if form.validate_on_submit():  # 내용 채우지 않은 항목이 있는지까지 체크
        usertable = Robot(robot_manufacture=None, robot_name=None, user_count_line_name=line, robot_id=None)
        usertable.robot_name = form.data.get('robot_name').strip()
        usertable.robot_manufacture = form.data.get('robot_manufacture').strip()
        db.session.add(usertable)  # DB저장
        db.session.commit()  # 변동사항 반영
        return redirect(url_for('dashboard_ad', id=id))
    return render_template('new_robot.html', form=form, line=line)


# 로봇 등록
@app.route('/dashboard/new_robot/', methods=['GET', 'POST'])
def new_robot():
    form = New_robotForm()
    user_line = Count.query.filter_by(user_id=session['user_id']).first()
    line = user_line.line_name
    if form.validate_on_submit():  # 내용 채우지 않은 항목이 있는지까지 체크
        usertable = Robot(robot_manufacture=None, robot_name=None, user_count_line_name=line, robot_id=None)
        usertable.robot_name = form.data.get('robot_name').strip()
        usertable.robot_manufacture = form.data.get('robot_manufacture').strip()
        db.session.add(usertable)  # DB저장
        db.session.commit()  # 변동사항 반영
        return redirect(url_for('dashboard', id=id))
    return render_template('new_robot.html', form=form, line=line)


# 센서 등록
@app.route('/dashboard/new_sensor/<int:robot_id>', methods=['GET', 'POST'])
def new_sensor(robot_id=None):
    robot = Robot.query.filter_by(robot_id=id).first()
    robot_name = robot.robot_name
    form = New_sensorForm()
    if form.validate_on_submit():  # 내용 채우지 않은 항목이 있는지까지 체크
        usertable = Sensor(sensor_name=None, axis=None, locate=None, robort_robot_id=id, sample=None)
        sensor_name = form.data.get('sensor_name').strip()
        usertable.sensor_name = sensor_name
        usertable.axis = form.data.get('axis').strip()
        usertable.locate = form.data.get('locate').strip()
        db.session.add(usertable)  # DB저장
        db.session.commit()  # 변동사항 반영
        return redirect(url_for('view_deteil_ad', robot_id=id))
    return render_template('new_sensor.html',
                           form=form,
                           robot_name=robot_name)


# 로봇 세부사항(view detail)
@app.route('/deteil/<int:robot_id>', methods=['GET'])
def view_deteil(robot_id=None):
    robot = Robot.query.filter_by(robot_id=robot_id).first()
    manufacture = robot.robot_manufacture
    line = robot.user_count_line_name
    robot_name = robot.robot_name
    id = robot.robot_id

    sensor = Sensor.query.filter_by(robot_robot_id=id).all()
    named = []
    locate = []
    axis = []
    result = []
    ai_result = []
    sensor_id = []
    for i in range(len(sensor)):
        name = sensor[i].sensor_name
        axis.append(sensor[i].axis)
        locate.append(sensor[i].locate)
        named.append(name)
        sensor_id.append(sensor[i].sensor_id)
        globals()['{}'.format(sensor[i].sensor_name)] = Schedata.query.filter_by(sensor_sensor_name=name).order_by(
            Schedata.data_id.desc())[:10]
        result.append(globals()['{}'.format(sensor[i].sensor_name)])
        globals()['{}'.format(sensor[i].sensor_name)] = Result.query.filter_by(sensor_sensor_name=name).order_by(
            Result.result_key.desc())[:10]
        ai_result.append(globals()['{}'.format(sensor[i].sensor_name)])

    print(sensor_id)
    Train.query.filter(Train.model_name == None).delete()
    db.session.commit()

    return render_template('view_details.html',
                           line=line,
                           named=named,
                           robot_name=robot_name,
                           result=result,
                           zip=zip,
                           axis=axis,
                           locate=locate,
                           ai_result=ai_result,
                           sensor_id=sensor_id,
                           id=id,
                           enumerate=enumerate,
                           manufacture=manufacture)


# 관리자 deteil
# 로봇 세부사항(view detail)
@app.route('/deteil_ad/<int:robot_id>', methods=['GET'])
def view_deteil_ad(robot_id=None):
    robot = Robot.query.filter_by(robot_id=robot_id).first()

    manufacture = robot.robot_manufacture
    line = robot.user_count_line_name
    robot_name = robot.robot_name

    id = robot.robot_id
    sensor = Sensor.query.filter_by(robot_robot_id=id).all()
    named = []
    locate = []
    axis = []
    result = []
    ai_result = []
    sensor_id = []
    for i in range(len(sensor)):
        name = sensor[i].sensor_name
        axis.append(sensor[i].axis)
        locate.append(sensor[i].locate)
        named.append(name)
        sensor_id.append(sensor[i].sensor_id)
        globals()['{}'.format(sensor[i].sensor_name)] = Schedata.query.filter_by(sensor_sensor_name=name).order_by(
            Schedata.data_id.desc())[:10]
        result.append(globals()['{}'.format(sensor[i].sensor_name)])
        globals()['{}'.format(sensor[i].sensor_name)] = Result.query.filter_by(sensor_sensor_name=name).order_by(
            Result.result_key.desc())[:10]
        ai_result.append(globals()['{}'.format(sensor[i].sensor_name)])
    return render_template('view_details_ad.html',
                           line=line,
                           named=named,
                           robot_name=robot_name,
                           result=result,
                           zip=zip,
                           axis=axis,
                           locate=locate,
                           ai_result=ai_result,
                           sensor_id=sensor_id,
                           enumerate=enumerate,
                           id=id,
                           manufacture=manufacture)


# scatter plot - sche data
@app.route('/plot/<int:sensor_id>', methods=['GET', 'POST'])
def plot_sensor(sensor_id=None):
    num = Train.query.filter_by(sensor_sensor_id=sensor_id).order_by(Train.num.desc()).first()
    sensor = Sensor.query.filter_by(sensor_id=sensor_id).first()
    robot_id = sensor.robot_robot_id
    sensor_name = sensor.sensor_name
    sensor_position = sensor.locate
    sensor_axis = sensor.axis
    sensor_sample = sensor.sample
    model_num = num.num
    data = Result.query.filter_by(train_data_num=model_num).all()
    data_ = data[-50:]
    sche_data = []
    predict = []
    score = []
    title = []
    for i in data_:
        predict.append(i.predict)
        score.append(i.anomal_score)
        sche = Schedata.query.filter_by(data_id=i.sche_data_data_id).first()
        x = pd.Series(np.fromstring(sche.data_set, dtype=float, sep=','))
        sche_data.append(x)
        title.append(sche.date_stamp)

    add_data = []
    add_title = []

    inlier, outlier = scatter(sche_data, predict)
    clicked = None
    if request.method == "POST":
        clicked = request.get_json()['data']
        clicked = int(clicked)

        add_data_ = sche_data[clicked]
        add_data = add_data_.tolist()

        add_title = title[clicked]

        return jsonify({
            'data': add_data,
            'title': add_title
        })

    return render_template('3charts.html',
                           add_data=add_data,
                           add_title=add_title,
                           robot_id=robot_id,
                           sensor_id=sensor_id,
                           sensor_name=sensor_name,
                           sensor_position=sensor_position,
                           sensor_axis=sensor_axis,
                           sensor_sample=sensor_sample,
                           inlier=inlier,
                           outlier=outlier,
                           score=score,
                           zip=zip,
                           enumerate=enumerate)


# scatter plot - train data
@app.route('/plot_train/<int:num>', methods=['GET', 'POST'])
def train_scatter(num=None):
    num = Train.query.filter_by(num=num).order_by(Train.num.desc()).first()
    sensor_id = num.sensor_sensor_id

    score = num.score
    predict = num.model_predict
    model_num = num.num
    sensor = Sensor.query.filter_by(sensor_id=sensor_id).first()
    data = pd.Series(np.fromstring(num.data_set, dtype=float, sep=','))
    score = pd.Series(np.fromstring(score, dtype=float, sep=','))
    predict = pd.Series(np.fromstring(predict, dtype=int, sep=','))
    predict = predict.tolist()
    score = score.tolist()

    data = np.array(data)
    sensor_name = num.sensor_sensor_name
    sensor_position = sensor.locate
    sensor_axis = sensor.axis
    sensor_sample = sensor.sample
    data = data.reshape(-1, sensor_sample)

    inlier, outlier = scatter(data, predict)

    return render_template('ex.html',
                           sensor_name=sensor_name,
                           sensor_position=sensor_position,
                           sensor_axis=sensor_axis,
                           sensor_sample=sensor_sample,
                           inlier=inlier,
                           outlier=outlier,
                           score=score,
                           zip=zip,
                           enumerate=enumerate, )


# train
@app.route('/train/<int:sensor_id>', methods=['GET', 'POST'])
def train(sensor_id=None):
    robot_id_ = Sensor.query.filter_by(sensor_id=sensor_id).first()
    robot_id = robot_id_.robot_robot_id
    sensor_name = robot_id_.sensor_name
    line_ = Robot.query.filter_by(robot_id=robot_id).first()
    line = line_.user_count_line_name
    robot_name = line_.robot_name
    train = Train.query.filter_by(sensor_sensor_id=sensor_id).order_by(Train.model_name.desc()).first()
    model_name = train.model_name
    date = train.date_stamp
    num = train.num

    sensor = Sensor.query.filter_by(robot_robot_id=robot_id).all()
    name = []
    sensor_id_ = []
    for i in sensor:
        name.append(i.sensor_name)
        sensor_id_.append(i.sensor_id)

    Train.query.filter(Train.model_name == None).delete()
    db.session.commit()

    return render_template('ai_center.html',
                           robot_id=robot_id,
                           num=num,
                           date=date,
                           model_name=model_name,
                           sensor_name=sensor_name,
                           robot_name=robot_name,
                           sensor_id=sensor_id_,
                           sen_id=sensor_id,
                           name=name,
                           zip=zip,
                           enumerate=enumerate
                           )


# train result
@app.route('/<int:sensor_id>,<start_date>,<end_date>', methods=['GET', 'POST'])
def train_result(sensor_id=None, start_date=None, end_date=None):
    start_date_ = start_date
    end_date_ = end_date
    sensor = Sensor.query.filter_by(sensor_id=sensor_id).first()
    sensor_name = sensor.sensor_name
    robot_id = sensor.robot_robot_id
    sensor_axis = sensor.axis
    print(sensor_axis)

    sensor = Sensor.query.filter_by(robot_robot_id=robot_id).all()
    name = []
    sensor_id_ = []
    for i in sensor:
        name.append(i.sensor_name)
        sensor_id_.append(i.sensor_id)
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    print(start_date)
    print(end_date)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("opcua").setLevel(logging.WARNING)

    # replace xx.xx.xx.xx with the IP address of your server
    serverIP = "25.27.135.161"
    serverUrl = urlparse('opc.tcp://{:s}:4840'.format(serverIP))

    # replace xx:xx:xx:xx with your sensors macId
    macId = sensor_name
    timeZone = "Asia/Seoul"  # local time zone
    now = start_date
    dt_s = now - datetime.timedelta(hours=24)
    dt_e = end_date
    startTime = dt_s.strftime('%Y-%m-%d 15:0:0')
    TIME = dt_e.strftime('%Y-%m-%d 15:0:0')
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
    # new train data up \load
    data, len_ = opc_ua.DataAcquisition.plotly(dates, values)

    title = []
    for i in range(len(dates)):
        title_ = datetime.datetime.strptime(dates[i], '%Y-%m-%d %H:%M:%S')
        title_ = title_.replace(tzinfo=pytz.timezone('UTC')).astimezone(pytz.timezone(timeZone))
        title.append(title_.strftime('%Y-%m-%dT%H:%M:%S'))
    form = ExampleForm()

    opc_ua.DataAcquisition.save_sql(data, sensor_name)

    data_ = []
    for i in range(len(data)):
        data_.append(list(map(float, data[i])))
    sensor_samples = Sensor.query.filter_by(sensor_name=sensor_name).first()
    sensor_samples.sample = len(data_[1])
    db.session.commit()
    num = Train.query.order_by(Train.num.desc()).first()
    last_num = num.num
    return render_template('train_set.html',
                           last_num=last_num,
                           sensor_name=sensor_name,
                           data=data_,
                           title=title,
                           name=name,
                           robot_id=robot_id,
                           sensor_=sensor_id,
                           sensor_id=sensor_id_,
                           form=form,
                           len=len_,
                           zip=zip,
                           enumerate=enumerate
                           )


# train resul2
@app.route('/train_/<int:sensor_id>', methods=['POST', 'GET'])
def date(sensor_id=None):
    sensor = Sensor.query.filter_by(sensor_id=sensor_id).first()
    robot_id = sensor.robot_robot_id

    sensor = Sensor.query.filter_by(robot_robot_id=robot_id).all()
    name = []
    sensor_id_ = []
    print(sensor_id_)
    for i in sensor:
        name.append(i.sensor_name)
        sensor_id_.append(i.sensor_id)
    form = ExampleForm()
    if form.validate_on_submit():
        start_date = form.dt.data.strftime("%Y-%m-%d")
        end_date = form.dta.data.strftime("%Y-%m-%d")
        return redirect(url_for('train_result', sensor_id=sensor_id, start_date=start_date, end_date=end_date))
    return render_template('train.html', form=form, name=name, sensor_id=sensor_id_, sensor_id_=sensor_id,
                           robot_id=robot_id, zip=zip)


# all dashboard
@app.route('/all', methods=['GET'])
def all_dash():
    line_ = Count.query.filter_by(factory_name=session['factory']).all()
    line = []
    alam = []
    for j in range(len(line_)):
        name = line_[j].line_name
        if len(Robot.query.filter_by(user_count_line_name=name).all()) == 0:
            continue
        line.append(name)
    for i in range(len(line)):
        globals()['{}'.format(line[i])] = Robot.query.filter_by(user_count_line_name=line[i]).all()
        alam.append(globals()['{}'.format(line[i])])
    print(line)
    for alam_el in alam:
        for j in range(len(alam_el)):
            id = alam_el[j].robot_id
            sensor = Sensor.query.filter_by(robot_robot_id=id).all()
            alam_el[j] = sensor
    for i in range(len(alam)):
        data = alam[i]
        for k in range(len(data)):
            data_ = data[k]
            for j in range(len(data_)):
                value = Result.query.filter_by(
                    sensor_sensor_name=data_[j].sensor_name).order_by(Result.result_key.desc()).first()
                if value == None:
                    continue
                elif value.predict == -1:
                    alam[i][k] = -1
                    break
                else:
                    alam[i][k] = 0
    robot_id_ = []
    first_robot = []
    for i in range(len(line)):
        globals()['{}'.format(line[i])] = Robot.query.filter_by(user_count_line_name=line[i]).all()
        robot_id_.append(globals()['{}'.format(line[i])])
        robot = Robot.query.filter_by(user_count_line_name=line[i]).first()
        first_robot.append(robot.robot_id)
    print(first_robot)
    return render_template('all_dash.html',
                           robot_id=robot_id_,
                           line=line,
                           alam=alam,
                           id=first_robot,
                           zip=zip,
                           enumerate=enumerate)


# model 생성및 저장
@app.route('/api/models/<int:num>', methods=['GET'])
def new_model(num=None):
    # model 생성및 저장
    item_ = Train.query.filter_by(num=num).first()
    item = item_.data_set
    sensor_name = item_.sensor_sensor_name
    sensor_id = item_.sensor_sensor_id
    # trainset testset 생성

    samples_ = Sensor.query.filter_by(sensor_name=sensor_name).all()

    samp = []
    for i in samples_:
        samp.append(i.sample)
    samp = int(samp[0])

    X_train, X_test = anomal_model.anomal_model_(item, detection_type, training_ratio, samp)
    print(X_train.shape)
    print(X_test.shape)

    # 본인data model score 생성
    model, score, result = anomal_model.algorithm_train(X_train, X_test)
    current_model_path = MODEL_FILE_PATH + format('svgmodel{}.model'.format(item_.num))
    model_write_file = open(current_model_path, "wb")
    joblib.dump(model, model_write_file)
    model_write_file.close()
    name = current_model_path.split('/')
    name = name[-1]
    score = score.tolist()
    score_ = []
    result_ = []
    for i in range(len(score)):
        score_.append(str(score[i]))
        result_.append(str(result[i]))

    score = ",".join(score_)
    result = ",".join(result_)
    item_.model_name = name
    item_.score = score
    item_.model_predict = result
    db.session.commit()
    print(current_model_path)
    gc.collect()

    return redirect(url_for('train', sensor_id=sensor_id))


@app.route('/modeling', methods=['POST', 'GET'])
def modeling():
    if request.method == 'POST':
        temp = request.form['id']

    else:
        temp = None
    return redirect(url_for('new_model', num=temp))


def scatter(data, predict):
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    # normalize the metrics
    X = scaler.fit_transform(data)
    X_train = scaler.fit_transform(data)
    X_reduce = pca.fit_transform(X)
    inlier = []
    outlier = []
    for i in range(len(predict)):
        if predict[i] == 1:
            inlier.append([X_reduce[i, 0], X_reduce[i, 1]])
        else:
            outlier.append([X_reduce[i, 0], X_reduce[i, 1]])
    return inlier, outlier


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5002", debug=True, use_reloader=False, threaded=True)
