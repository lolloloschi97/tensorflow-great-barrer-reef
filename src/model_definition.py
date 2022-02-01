from hyper_param import *

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='gpu', classes = 1, autoshape = False)  # load on GPU