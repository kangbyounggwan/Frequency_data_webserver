from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, Text, JSON, DateTime, LargeBinary, Boolean, Column
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
import os
from flask_caching import Cache
from flask import Flask

app = Flask(__name__)
cache = Cache(app,config={'CACHE_TYPE': 'simple'})


app.debug = True
# Flask Setting
SECRET_KEY = '\xb0\xdbh\x12\x0f\xab\xed{}\xad^\xfa\xfa/\xd3\xf5\xd5\x13\x9f\x1f\xc3<\xd6'
FLASK_ADMIN = 'Servo Gun MLServer'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = SECRET_KEY
app.config['FLASK_ADMIN'] = FLASK_ADMIN
app.config['CHARSET'] = 'utf-8'
app.config['JSON_AS_ASCII'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:tachyon123@kbg.co7hg2djahjf.ap-northeast-2.rds.amazonaws.com:3306/dongseo'
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_size': 100, 'pool_recycle': 280}
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:tachyon123@34.64.188.127:3306/dongseo'
app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False

from flask_debugtoolbar import DebugToolbarExtension

toolbar = DebugToolbarExtension(app)

# from .admin import create_admin
# admin = create_admin(app)
db = SQLAlchemy(app)
cached = Cache(app)
migrate = Migrate(app, db)
ma = Marshmallow(app)


class Result(db.Model):
    __tablename__ = 'result'
    date_stamp = db.Column('date_stamp', DateTime)
    anomal_score = db.Column('anomal_score', Integer)
    predict = db.Column('predict', Integer)
    result_key = db.Column('result_key', Integer, primary_key=True)
    train_data_num = db.Column('train_data_num', Integer, db.ForeignKey('train_data.num'))
    sche_data_data_id = db.Column('sche_data_data_id', Integer, db.ForeignKey('sche_data.data_id'))
    sensor_sensor_name = db.Column('sensor_sensor_name', String(20), db.ForeignKey('sensor.sensor_name'))
    sensor_sensor_id = db.Column('sensor_sensor_id', Integer, db.ForeignKey('sensor.sensor_id'))

    def __init__(self, anomal_score, date_stamp, predict, result_key, train_data_num, sche_data_data_id,
                 sensor_sensor_name, sensor_sensor_id):
        self.anomal_score = anomal_score
        self.date_stamp = date_stamp
        self.predict = predict
        self.result_key = result_key
        self.train_data_num = train_data_num
        self.sche_data_data_id = sche_data_data_id
        self.sensor_sensor_name = sensor_sensor_name
        self.sensor_sensor_id = sensor_sensor_id


class ResultSchema(ma.Schema):
    class Meta:
        fields = (' anomaly_score', 'date_stamp', 'predict', 'result_key', 'train_data_num', 'train_data_model_name',
                  'sensor_sensor_id')


result_schema = ResultSchema()
result_schema_ = ResultSchema(many=True)


class Robot(db.Model):
    __tablename__ = 'robot'
    robot_id = db.Column('robot_id', Integer, primary_key=True)
    robot_name = db.Column('robot_name', String(30))
    robot_manufacture = db.Column('robot_manufacture', String(20))
    user_count_line_name = db.Column('user_count_line_name', String(50), db.ForeignKey('user_count.line_name'))

    def __init__(self, robot_id, robot_name, robot_manufacture, user_count_line_name):
        self.robot_id = robot_id
        self.robot_name = robot_name
        self.robot_manufacture = robot_manufacture
        self.user_count_line_name = user_count_line_name


class RobotSchema(ma.Schema):
    class Meta:
        fields = (' robot_id', 'robot_name', 'robot_manufacture', 'user_count_line_name')


robot_schema = RobotSchema()
robot_schema_ = RobotSchema(many=True)


class Schedata(db.Model):
    __tablename__ = 'sche_data'
    date_stamp = db.Column('date_stamp', DateTime)
    data_set = db.Column('data_set', Text)
    data_id = db.Column('data_id', Integer, primary_key=True)
    sensor_sensor_name = db.Column('sensor_sensor_name', String(20), db.ForeignKey('sensor.sensor_name'))
    sensor_sensor_id = db.Column('sensor_sensor_id', Integer, db.ForeignKey('sensor.sensor_id'))

    def __init__(self, date_stamp, data_set, data_id, sensor_sensor_name, sensor_sensor_id):
        self.date_stamp = date_stamp
        self.data_set = data_set
        self.data_id = data_id
        self.sensor_sensor_name = sensor_sensor_name
        self.sensor_sensor_id = sensor_sensor_id


class Sche_dataSchema(ma.Schema):
    class Meta:
        fields = ('date_stamp', 'data_set', 'data_id', 'sensor_sensor_name', 'sensor_sensor_id')


sche_schema = Sche_dataSchema()
sche_schema_ = Sche_dataSchema(many=True)


class Sensor(db.Model):
    __tablename__ = 'sensor'
    sensor_name = db.Column('sensor_name', String(20), primary_key=True)
    axis = db.Column('axis', String(5))
    robot_robot_id = db.Column('robot_robot_id', Integer, db.ForeignKey('robot.robot_id'))
    locate = db.Column('locate', String(20))
    sample = db.Column("sample", Integer)
    sensor_id = db.Column('sensor_id', Integer, primary_key=True)

    def __init__(self, sensor_name, axis, robort_robot_id, locate, sample):
        self.sensor_name = sensor_name
        self.axis = axis
        self.robot_robot_id = robort_robot_id
        self.locate = locate
        self.sample = sample


class SensorSchema(ma.Schema):
    class Meta:
        fields = ('sensor_name', ' axis', 'robort_robot_id', "sample", 'sensor_id')


sensor_schema = SensorSchema()
sensor_schema_ = SensorSchema(many=True)


class Train(db.Model):
    __tablename__ = 'train_data'
    date_stamp = db.Column('date_stamp', DateTime)
    data_set = db.Column('data_set', Text)
    model_name = db.Column('model_name', String(20))
    sensor_sensor_name = db.Column('sensor_sensor_name', Integer, db.ForeignKey('sensor.sensor_name'))
    sensor_sensor_id = db.Column('sensor_sensor_id', Integer, db.ForeignKey('sensor.sensor_id'))
    num = db.Column('num', Integer, primary_key=True)
    model_predict = db.Column('model_predict', Text)
    score = db.Column('score', Text)

    def __init__(self, date_stamp, data_set, model_name, sensor_sensor_name, sensor_senor_id, num, model_predict,
                 score):
        self.date_stamp = date_stamp
        self.data_set = data_set
        self.model_name = model_name
        self.sensor_sensor_name = sensor_sensor_name
        self.sensor_sensor_id = sensor_senor_id
        self.num = num
        self.model_predict = model_predict
        self.score = score


class TrainSchema(ma.Schema):
    class Meta:
        fields = (
            ' date_stamp', 'data_set', 'model_name', 'sensor_sensor_name', 'sensor_sensor_id', 'num', 'model_predict',
            'score')


train_schema = TrainSchema()
train_schema_ = TrainSchema(many=True)


class Count(db.Model):
    __tablename__ = 'user_count'
    user_id = db.Column('user_id', String(20))
    factory_name = db.Column('factory_name', String(20))
    line_name = db.Column('line_name', String(50), primary_key=True)
    password = db.Column('password', String(20))
    user_name = db.Column('user_name', String(10))
    approval_status = db.Column('approval_status', String(10))

    def __init__(self, user_id, factory_name, line_name, password, user_name, approval_status):
        self.user_id = user_id
        self.factory_name = factory_name
        self.line_name = line_name
        self.password = password
        self.user_name = user_name
        self.approval_status = approval_status


class CountSchema(ma.Schema):
    class Meta:
        fields = (' user_id', 'factory_name', 'line_name', 'password', 'user_name', 'approval_status')


count_schema = CountSchema()
count_schema_ = CountSchema(many=True)


class Ad_count(db.Model):
    __tablename__ = 'admin_count'
    user_id = db.Column('user_id', String(20), primary_key=True)
    factory_name = db.Column('factory_name', String(20))
    password = db.Column('password', String(20))
    user_name = db.Column('user_name', String(10))

    def __init__(self, user_id, factory_name, password, user_name):
        self.user_id = user_id
        self.factory_name = factory_name
        self.password = password
        self.user_name = user_name


class Ad_CountSchema(ma.Schema):
    class Meta:
        fields = (' user_id', 'factory_name', 'password', 'user_name')


ad_count_schema = Ad_CountSchema()
ad_count_schema_ = Ad_CountSchema(many=True)
