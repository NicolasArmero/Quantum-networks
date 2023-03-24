import socket
import pickle

msg = "Hello Client"
fallo = 1
while fallo == 1:
    try:
        port = int(input("Insert port: "))
        fallo = 0
        print("Server up and listening")
        bytes_tx = str.encode(msg)

        socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        socket.bind(("127.0.0.1", port))
        while True:
            bytes_rx = socket.recvfrom(1024)
            unwrap = pickle.loads(bytes_rx[0])
            print(unwrap)
            id = unwrap[1]
            message = unwrap[0]
            num = len(message)
            address = bytes_rx[1]
            print("Message received: ",message)
            print(f"Characteres received: {num}")
            print(f"Client {id}" + " IP address: ",address[0] + "  and port: ",str(address[1]))
            socket.sendto(bytes_tx, address)
            if "EXIT" in message.upper():
                socket.close()
                break
                
    except ValueError:
        print("Must be a number")
        fallo = 1
    
    print("Server closed succesfully")