import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from robot_control.commands import RobotCommand

class _CmdVelNode(Node):
    def __init__(self, node_name: str, topic: str):
        super().__init__(node_name)
        self.pub = self.create_publisher(Twist, topic, 10)

    def send_twist(self, vx=0.0, vy=0.0, wz=0.0):
        msg = Twist(); msg.linear.x=vx; msg.linear.y=vy; msg.angular.z=wz
        self.pub.publish(msg)

class ROS2CmdVelClient:
    def __init__(self, robot_id: str, topic: str="/cmd_vel", node_name: str|None=None):
        self.robot_id = robot_id
        self.topic = topic
        self.node_name = node_name or f"cmdvel_{robot_id}"
        self._node = None
        self._spin_th = None
        self._ensure_started()

    def _ensure_started(self):
        if self._node: return
        rclpy.init(args=None)
        self._node = _CmdVelNode(self.node_name, self.topic)
        self._spin_th = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_th.start()

    def send(self, cmd: RobotCommand):
        if not self._node: return
        n = (cmd.name or "").upper()
        p = cmd.params or {}
        if n == "STOP":
            self._node.send_twist(0.0, 0.0, 0.0)
        elif n == "MOVE":
            self._node.send_twist(float(p.get("vx",0.0)), float(p.get("vy",0.0)), float(p.get("wz",0.0)))
        elif n == "TURN":
            self._node.send_twist(0.0, 0.0, float(p.get("wz",0.0)))
        # else: ignora/custom
