# -*- coding: utf-8 -*-

import warnings

from ultralytics.data import YOLODataset

warnings.filterwarnings('ignore')
from ultralytics import YOLO
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # model.load('normal.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    #model = YOLO(model=r'D:\graduation project\ultralytics-main\normal.pt')
    model = YOLO(model=r'normal.pt')
import yaml

#yaml_path = r"D:\graduation project\ultralytics-main\Argoverse.yaml"
yaml_path = r"Argoverse.yaml"
try:
    # model.train(data=r'Argoverse.yaml',
    #             imgsz=500,
    #             epochs=300,
    #             batch=8,
    #             workers=1,
    #             device='',
    #             optimizer='SGD',
    #             close_mosaic=10,
    #             resume=False,
    #             project='runs/train',
    #             name='exp',
    #             single_cls=False,
    #             cache=False,
    #             )

    model.train(data=r'Argoverse.yaml',
                imgsz=640,
                epochs=300,
                batch=32,  # 每个GPU的批次大小
                workers=8,  # 数据加载线程数
                device='0,1,2,3',  # 使用4块GPU
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp_4gpu',  # 实验名称
                single_cls=False,
                cache=True,  # 启用数据集缓存
                )


except Exception as e:
    print(f"训练过程中发生错误: {e}")