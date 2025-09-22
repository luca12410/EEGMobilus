import os, socket, json, rclpy, sys
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

HOST = os.getenv("UDP_HOST", "0.0.0.0")
PORT = int(os.getenv("UDP_PORT", "9999"))
TOPIC = os.getenv("CMDVEL_TOPIC", "/cmd_vel")

class UdpCmdVelBridge(Node):
    def __init__(self):
        super().__init__("udp_cmdvel_bridge")
        dom = os.getenv("ROS_DOMAIN_ID", "<unset>")
        rmw = os.getenv("RMW_IMPLEMENTATION", "<unset>")
        self.get_logger().info(f"Bridge UP | UDP {HOST}:{PORT} -> ROS2 {TOPIC} | DOMAIN={dom} RMW={rmw}")

        qos = QoSProfile(depth=10)
        self.pub = self.create_publisher(Twist, TOPIC, qos)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((HOST, PORT))
        self.sock.setblocking(False)

        self.n_rx = 0
        self.n_pub = 0
        self.create_timer(0.01, self.loop)              
        self.create_timer(1.0, self._heartbeat)          
        self.create_timer(1.0, self._stats)

    def _heartbeat(self):
        self.pub.publish(Twist())
        self.n_pub += 1
        self.get_logger().debug("heartbeat")

    def _stats(self):
        self.get_logger().info(f"stats: rx/s={self.n_rx} pub/s={self.n_pub}")
        self.n_rx = 0; self.n_pub = 0

    def loop(self):
        try:
            data, _ = self.sock.recvfrom(65535)
        except BlockingIOError:
            return
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception as e:
            self.get_logger().warn(f"bad json: {e}")
            return
        self.n_rx += 1
        p = msg.get("params", {}) or {}
        tw = Twist()
        tw.linear.x  = float(p.get("vx", 0.0))
        tw.linear.y  = float(p.get("vy", 0.0))
        tw.angular.z = float(p.get("wz", 0.0))
        self.pub.publish(tw)
        self.n_pub += 1
        self.get_logger().info(f"rx→pub: vx={tw.linear.x:.3f} wz={tw.angular.z:.3f}")

def main():
    rclpy.init()
    node = UdpCmdVelBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
