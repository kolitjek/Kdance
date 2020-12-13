import socket
import ast
import numpy as np

class UDP:
    UDP_IP = "192.168.0.11"  # Has to be send on the client side (sender)
    UDP_port = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class Sender(UDP):
    def __init__(self):
        print("Sending to UDP target IP: %s" % self.UDP_IP)
        print("Sending to UDP target port: %s" % self.UDP_port)

    def broadcast(self, predictions=[], skeleton_data=[]):
        composed_list = [predictions, skeleton_data]
        self.sock.sendto(bytes(str(composed_list).encode('utf-8')), (self.UDP_IP, self.UDP_port))


class Receiver(UDP):
    def __init__(self):
        hostname = socket.gethostname()
        #self.UDP_IP = socket.gethostbyname(hostname)
        print("Host name: ", hostname)
        print("Listening to UDP target IP: %s" % self.UDP_IP)
        print("Listening to UDP target port: %s" % self.UDP_port)

        ## getting the IP address using socket.gethostbyname() method
        # self.UDP_IP = socket.gethostbyname(hostname)
        self.sock.bind((self.UDP_IP, self.UDP_port))

    def listen (self):
        while True:
            data, addr = self.sock.recvfrom(2048)

            res = ast.literal_eval(data.decode("utf-8"))
            return res
