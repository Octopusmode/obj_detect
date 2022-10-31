from ast import pattern
import json, os


def init(file_path='config.json', pattern_id=0):
    default_data = {"default":
                        {"polygons": (((0,0), (1040,0), (1030, 520), (445, 1080), (0, 1080)),
                                      ((100, 100), (1300, 100), (800, 600))),
                         "model_path": 'dnn_model/',
                         "data_path": 'resources/',
                         "frame_size": (320, 320),
                         }}
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(default_data, f, indent=2)
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        with open(file_path, "r") as f:
            data = json.load(f)
    return data