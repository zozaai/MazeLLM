from dataclasses import dataclass

@dataclass
class Position:
    x: int = 0
    y: int = 0
  
class Robot:
    def __init__(self):
        self.position = Position(x=0, y=0)


    
