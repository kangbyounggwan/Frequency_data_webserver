import time
import sys
import pytz
import logging
import itertools
from urllib.parse import urlparse
import pandas as pd
from opcua import ua, Client
import pymysql
import datetime
from models import Schedata, db, Train, Result, Sensor

conn = pymysql.connect(host='kbg.co7hg2djahjf.ap-northeast-2.rds.amazonaws.com', user='root', passwd='tachyon123', db='dongseo', charset='utf8')


class OpcUaClient(object):
    CONNECT_TIMEOUT = 15  # [sec]
    RETRY_DELAY = 10  # [sec]
    MAX_RETRIES = 3  # [-]

    class Decorators(object):
        @staticmethod
        def autoConnectingClient(wrappedMethod):
            def wrapper(obj, *args, **kwargs):
                for retry in range(OpcUaClient.MAX_RETRIES):
                    try:
                        return wrappedMethod(obj, *args, **kwargs)
                    except ua.uaerrors.BadNoMatch:
                        raise
                    except Exception:
                        pass
                    try:
                        obj._logger.warn('(Re)connecting to OPC-UA service.')
                        obj.reconnect()
                    except ConnectionRefusedError:
                        obj._logger.warn(
                            'Connection refused. Retry in 10s.'.format(
                                OpcUaClient.RETRY_DELAY
                            )
                        )
                        time.sleep(OpcUaClient.RETRY_DELAY)
                else:  # So the exception is exposed.
                    obj.reconnect()
                    return wrappedMethod(obj, *args, **kwargs)

            return wrapper

    def __init__(self, serverUrl):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = Client(
            serverUrl.geturl(),
            timeout=self.CONNECT_TIMEOUT
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        self._client = None

    @property
    @Decorators.autoConnectingClient
    def sensorList(self):
        return self.objectsNode.get_children()

    @property
    @Decorators.autoConnectingClient
    def objectsNode(self):
        path = [ua.QualifiedName(name='Objects', namespaceidx=0)]
        return self._client.get_root_node().get_child(path)

    def connect(self):
        self._client.connect()
        self._client.load_type_definitions()

    def disconnect(self):
        try:
            self._client.disconnect()
        except Exception:
            pass

    def reconnect(self):
        self.disconnect()
        self.connect()

    @Decorators.autoConnectingClient
    def get_browse_name(self, uaNode):
        return uaNode.get_browse_name()

    @Decorators.autoConnectingClient
    def get_node_class(self, uaNode):
        return uaNode.get_node_class()

    @Decorators.autoConnectingClient
    def get_namespace_index(self, uri):
        return self._client.get_namespace_index(uri)

    @Decorators.autoConnectingClient
    def get_child(self, uaNode, path):
        return uaNode.get_child(path)

    @Decorators.autoConnectingClient
    def read_raw_history(self,
                         uaNode,
                         starttime=None,
                         endtime=None,
                         numvalues=0,
                         cont=None):
        details = ua.ReadRawModifiedDetails()
        details.IsReadModified = False
        details.StartTime = starttime or ua.get_win_epoch()
        details.EndTime = endtime or ua.get_win_epoch()
        details.NumValuesPerNode = numvalues
        details.ReturnBounds = True
        result = OpcUaClient._history_read(uaNode, details, cont)
        assert (result.StatusCode.is_good())
        return result.HistoryData.DataValues, result.ContinuationPoint

    @staticmethod
    def _history_read(uaNode, details, cont):
        valueid = ua.HistoryReadValueId()
        valueid.NodeId = uaNode.nodeid
        valueid.IndexRange = ''
        valueid.ContinuationPoint = cont

        params = ua.HistoryReadParameters()
        params.HistoryReadDetails = details
        params.TimestampsToReturn = ua.TimestampsToReturn.Both
        params.ReleaseContinuationPoints = False
        params.NodesToRead.append(valueid)
        result = uaNode.server.history_read(params)[0]
        return result


class DataAcquisition(object):
    LOGGER = logging.getLogger('DataAcquisition')
    AXES = ('x', 'y', 'z')
    ORDINATES = ('accel', 'veloc')
    DOMAINS = ('time', 'freq')
    MAX_VALUES_PER_ENDNODE = 100  # Num values per endnode
    MAX_VALUES_PER_REQUEST = 2  # Num values per history request

    @staticmethod
    def selected_to_workbook(serverUrl,
                             macIdsToCollect,
                             starttime,
                             endtime):
        with OpcUaClient(serverUrl) as client:
            for sensorNode in client.sensorList:
                assert (client._client.uaclient._uasocket.timeout == 15)
                macId = client.get_browse_name(sensorNode).Name
                if macId not in macIdsToCollect:
                    DataAcquisition.LOGGER.info(
                        'Skipping sensor {:s}'.format(macId)
                    )
                    continue
                tagPath = ua.QualifiedName(
                    'deviceTag',
                    sensorNode.nodeid.NamespaceIndex
                )
                DataAcquisition.LOGGER.info(
                    'Processing sensor {:s} ({:s})'.format(
                        macId,
                        client.get_child(sensorNode, tagPath).get_value()
                    )
                )
                DataAcquisition.get_sensor_data(
                    client,
                    sensorNode,
                    starttime,
                    endtime
                )

    @staticmethod
    def get_sensor_data(serverUrl, macId, browseName, starttime, endtime, axis_):
        allValues = []
        allDates = []
        with OpcUaClient(serverUrl) as client:
            assert (client._client.uaclient._uasocket.timeout == 15)
            sensorNode = DataAcquisition.get_sensor_node(
                client,
                macId,
                browseName
            )

            for path in DataAcquisition.endnodes_path_generator(sensorNode, axis_):
                DataAcquisition.LOGGER.info(
                    'Browsing {:s} -> {:s}'.format(
                        macId,
                        sensorNode.get_browse_name().Name
                    )
                )
                endNode = client.get_child(sensorNode, path)
                (values, dates) = DataAcquisition.get_endnode_data(
                    client,
                    endNode,
                    starttime,
                    endtime
                )
                allValues.extend(values)
                allDates.extend(dates)
        return (allValues, allDates)

    @staticmethod
    def endnodes_path_generator(sensorNode, axis_):
        for (axis, ordinate, domain) in \
                itertools.product(DataAcquisition.AXES,
                                  DataAcquisition.ORDINATES,
                                  DataAcquisition.DOMAINS):
            if ordinate == 'accel' and domain == 'freq' and axis == axis_:
                # browseName: e.g. xAccelTime
                browseName = ''.join([
                    axis, ordinate.capitalize(), domain.capitalize()
                ])

                nsIdx = sensorNode.nodeid.NamespaceIndex  # iQunet namespace index
                path = [
                    ua.QualifiedName(axis, nsIdx),  # e.g. 'x'
                    ua.QualifiedName(ordinate, nsIdx),  # e.g. 'accel'
                    ua.QualifiedName(browseName, nsIdx),  # e.g. 'xAccelTime'
                ]
                print(ordinate, ':', domain, ':', axis)
                yield path

    @staticmethod
    def get_sensor_node(client, macId, browseName):
        nsIdx = client.get_namespace_index(
            'http://www.iqunet.com'
        )  # iQunet namespace index
        bpath = [
            ua.QualifiedName(name=macId, namespaceidx=nsIdx),
            ua.QualifiedName(name=browseName, namespaceidx=nsIdx)
        ]
        sensorNode = client.objectsNode.get_child(bpath)
        return sensorNode

    @staticmethod
    def get_endnode_data(client, endNode, starttime, endtime):
        dvList = DataAcquisition.download_endnode(
            client,
            endNode,
            starttime,
            endtime
        )
        dates, values = ([], [])
        for dv in dvList:
            dates.append(dv.SourceTimestamp.strftime('%Y-%m-%d %H:%M:%S'))
            values.append(dv.Value.Value.y_ordinate)

        # If no starttime is given, results of read_raw_history are reversed.
        if starttime is None:
            values.reverse()
            dates.reverse()
        return (values, dates)

    @staticmethod
    def download_endnode(client, endNode, starttime, endtime):
        endNodeName = client.get_browse_name(endNode).Name
        DataAcquisition.LOGGER.info(
            'Downloading endnode {:s}'.format(
                endNodeName
            )
        )
        dvList, contId = [], None
        while True:
            remaining = DataAcquisition.MAX_VALUES_PER_ENDNODE - len(dvList)
            assert (remaining >= 0)
            numvalues = min(DataAcquisition.MAX_VALUES_PER_REQUEST, remaining)
            partial, contId = client.read_raw_history(
                uaNode=endNode,
                starttime=starttime,
                endtime=endtime,
                numvalues=numvalues,
                cont=contId
            )
            if not len(partial):
                DataAcquisition.LOGGER.warn(
                    'No data was returned for {:s}'.format(endNodeName)
                )
                break
            dvList.extend(partial)
            sys.stdout.write('\r    Loaded {:d} values, {:s} -> {:s}'.format(
                len(dvList),
                str(dvList[0].ServerTimestamp.strftime("%Y-%m-%d %H:%M:%S")),
                str(dvList[-1].ServerTimestamp.strftime("%Y-%m-%d %H:%M:%S"))
            ))
            sys.stdout.flush()
            if contId is None:
                break  # No more data.
            if len(dvList) >= DataAcquisition.MAX_VALUES_PER_ENDNODE:
                break  # Too much data.
        sys.stdout.write('...OK.\n')

        return dvList

    @staticmethod
    def plotly(dates, values):
        len_ = []
        datalist = []
        for i in range(len(dates)):
            len_.append(len(values[0][30:]))
            datalist.append(values[i][30:])
        return datalist, len_

    @staticmethod
    # train data
    def save_sql(data, name):

        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime)
        raw = []
        for i in range(len(data)):
            d = data[i]
            for j in range(len(d)):
                d[j] = str(d[j])
                raw.append(d[j])
        row_str = ",".join(raw)

        sensor_id = Sensor.query.filter_by(sensor_name=name).first()
        datafild = Train(date_stamp=None, data_set=None, model_name=None, sensor_senor_id=None, num=None,
                         sensor_sensor_name=None, model_predict=None, score=None)

        datafild.date_stamp = datetime.datetime.now()
        datafild.data_set = row_str
        datafild.sensor_sensor_name = name
        datafild.sensor_sensor_id = sensor_id.sensor_id
        db.session.add(datafild)  # DB저장
        db.session.commit()
        db.session.expire(datafild)
        db.session.refresh(datafild)

    @staticmethod
    # test_data
    def sche(data, title, result, score, sensor_name, model, num):
        raw = []
        print(title)
        data = data[-1]
        score = score[0]
        print(score)
        result = result[0]
        for j in range(len(data)):
            data[j] = str(data[j])
            raw.append(data[j])

        row_str = ",".join(raw)
        sensor_id = Sensor.query.filter_by(sensor_name=sensor_name).first()
        sen_id = sensor_id.sensor_id

        sql = 'insert into sche_data(date_stamp,data_set,sensor_sensor_id,sensor_sensor_name) values(%s, %s,%s,%s)'
        curs = conn.cursor()
        curs.execute(sql, (title, row_str, sen_id, sensor_name))
        conn.commit()



        curs.execute('SELECT * FROM dongseo.sche_data where date_stamp = %s', title)
        date_raw = curs.fetchall()


        id = date_raw[0][2]
        print(id)
        # data_result = Result(date_stamp=None, anomal_score=None, predict=None, result_key=None, train_data_num=None,
        #                      sche_data_data_id=id, sensor_sensor_name=None, sensor_sensor_id=sen_id)
        # data_result.date_stamp = datetime.datetime.now()
        # data_result.anomal_score = score
        # data_result.predict = result
        # data_result.train_data_num = num
        # data_result.sensor_sensor_name = sensor_name
        # db.session.add(data_result)
        # db.session.commit()  # 변동사항 반영
        # db.session.refresh(data_result)  # DB저장
        sql = 'insert into result(date_stamp, anomal_score, predict, train_data_num, sche_data_data_id, sensor_sensor_name, sensor_sensor_id) values(%s,%s,%s,%s,%s,%s,%s)'
        curs.execute(sql, (datetime.datetime.now(), score, result, num,id,sensor_name,sen_id))
        conn.commit()

        return title
