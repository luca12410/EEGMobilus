from dataclasses import dataclass, field
from typing import Dict, Any, Optional

CMD_MOVE   = "MOVE"
CMD_TURN   = "TURN"
CMD_STOP   = "STOP"
CMD_CUSTOM = "CUSTOM"

@dataclass
class RobotCommand:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    target: Optional[str] = None