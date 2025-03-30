# -*- coding: utf-8 -*-
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\graduation project\ultralytics-main\best.pt')
    model.predict(source=r'D:\graduation project\ultralytics-main\datasets\Argoverse\images\val1',
                  save=True,
                  show=True,
                  )
