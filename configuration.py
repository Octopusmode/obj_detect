from ast import pattern
import json, os


def init(file_path='config.json', pattern_id=0):
    default_data = {"default":
                        {"polygons": (((0, 0.17), (0.13, 0.15), (0.54, 0.48), (0.23, 1), (0, 1)),
                                      ((0.83, 0.09), (0.99, 0.09), (0.99, 0.83), (0.83, 0.83))),
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