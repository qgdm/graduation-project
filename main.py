import deep_sort_pytorch as deepsort
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(model=r'D:\graduation project\ultralytics-main\yolo11n.pt')
    results=model.predict(source=r'D:\graduation project\ultralytics-main\ultralytics\assets',
                  save=True,
                  show=True,
                  )
