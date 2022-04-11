import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
recv_address = ('127.0.0.1', 9999)
sock.bind(recv_address)

sock.listen(1)

conn, addr = sock.accept()

# recv and send loop
while True:
    data = conn.recv(655525)
    # 받고 data를 돌려줌.
    print('기강 잡는 그남자:', data.decode())
    if data != None:
        go_cl = input(str('나:'))
        conn.send(go_cl.encode())

conn.close()