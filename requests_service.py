# -*- coding:utf-8 -*-
import requests

#모델 생성 완료 시 서버에 임계치 저장 요청
def model_complete(id):
    URL = 'http://127.0.0.1:5001/restApi/servogunmaster/curupdate/' + str(id)
    #headers = {'Content-Type': 'application/json; charset=utf-8'}
    print(URL)
    res = requests.get(URL)
    print(res)