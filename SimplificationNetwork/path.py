import json


class Path:
    def __init__(self, path_file):
        with open(path_file, 'r') as f:
            self.__dict__ = json.loads(f.read())

    def get_scaled_path(self, scale):
        scaled = []
        for group in self.groups:
            for stroke in group["strokes"]:
                points = []
                for point in stroke["points"]:
                    points.append([scale[0] * point["x"], scale[1] * point["y"]])
                scaled.append(points)
        return scaled
