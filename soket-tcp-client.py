import socket
import random
import pickle
import time
import numpy as np

# socket create and connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 30010))


# send msg
while(1):
    test = np.random.rand(4000)
    data_ = test.tolist()
    data = pickle.dumps(data_)
    print(test)

    sock.send(data)
    time.sleep(0.25)
# connection close
sock.close()