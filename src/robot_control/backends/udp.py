import socket, json

class UDPClient:
    def __init__(self, host="127.0.0.1", port=9999):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd):
        payload = {"name": cmd.name, "params": cmd.params, "target": cmd.target}
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
