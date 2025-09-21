from typing import Dict, Optional
from robot_control.commands import RobotCommand

class RobotRouter:
    def __init__(self): self._clients: Dict[str, object] = {}
    def register(self, robot_id: str, client: object): self._clients[robot_id] = client
    def dispatch(self, cmd: RobotCommand, default_robot: Optional[str]=None):
        rid = cmd.target or default_robot
        if rid:
            if rid in self._clients: self._clients[rid].send(cmd)
        else:
            for c in self._clients.values(): c.send(cmd)  # broadcast
