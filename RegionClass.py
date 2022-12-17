class Region:
    def __init__(self, id, centr):
        self.id = id
        self.centr = centr
        self.points = []
        self.points_cord = []
        self.type = 0
        self.neighbors = []
        self.wetness = 0
        self.height = 0
        self.temperature = 0
        self.rivers = False
    
    def to_json(self):
        fields = {
            'id': self.id,
            'centr': self.centr,
            'points': self.points,
            'points_cord': self.points_cord,
            'type': self.type,
            'neighbors': self.neighbors,
            'wetness': self.wetness,
            'height': self.height,
            'temperature': self.temperature,
            'rivers': self.rivers
        }
        return fields
