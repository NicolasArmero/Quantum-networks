import socket
import pickle
import time

from sys import argv

#msg = "Hello Server, I'm Client 1"
msg = input("Insert message: ")
id = argv[3]
men = (msg, id)
bytes_tx = pickle.dumps(men)

fallo = 1
while fallo == 1:
    try:
        ip = argv[1]
        port = int(argv[2])
        fallo = 0
        server_address = (ip, port)
        soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        try:
            print("Trying to connect")
            soc.sendto(bytes_tx, server_address)
            bytes_rx = soc.recvfrom(1024)
            print("RX: ",bytes_rx)
        except:
            try:
                print("Trying to connect again in 10 s")
                time.sleep(10)
                soc.sendto(bytes_tx, server_address)
                bytes_rx = soc.recvfrom(1024)
                print("RX: ",bytes_rx)
            except:
                print("Connection time ran out")
        soc.close()

    except ValueError:
        print("Must be a number")
        fallo = 1