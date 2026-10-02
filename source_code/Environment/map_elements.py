class Checkpoint:
    """
    This class represents a checkpoint in the sailing environment.
    Each checkpoint has a position, a number (identifier and the position in the race), and a radius that defines the area around the checkpoint that the boat
    must enter to be considered as having reached the checkpoint.
    """
    def __init__(self, position, number, radius):
        self.position = position
        self.number = number
        self.radius = radius
