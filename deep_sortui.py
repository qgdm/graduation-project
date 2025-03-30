import torch
import cv2
import numpy as np
import os
import sys
import logging

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler('app.log'),
                            logging.StreamHandler()])

try:
    # 解决 OpenMP 冲突
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    logging.info("环境变量设置成功")
except Exception as e:
    logging.error(f"环境变量设置失败: {str(e)}")

# 仅保留车辆和行人类别
CLASS_NAMES = {
    0: "person",
    2: "car"
}

# 颜色映射
CLASS_COLORS = {
    0: (0, 255, 0),  # 绿色：行人
    2: (255, 0, 0)   # 蓝色：车辆
}

# 初始化YOLO模型
def load_yolo(model_path, device):
    try:
        model = YOLO(model_path, task="detect")  # 显式指定任务
        logging.info(f"YOLO模型加载成功: {model_path}")
        return model
    except Exception as e:
        logging.error(f"YOLO模型加载失败: {str(e)}")
        raise

# 初始化Deep SORT
def init_deepsort():
    try:
        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        
        # 修改模型文件路径
        cfg.DEEPSORT.REID_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "deep_sort_pytorch", "deep_sort", "deep", "checkpoint", "ckpt.t7")
        
        # 确保REID_CKPT路径存在
        if not os.path.exists(cfg.DEEPSORT.REID_CKPT):
            logging.error(f"DeepSORT模型文件不存在: {cfg.DEEPSORT.REID_CKPT}")
            raise FileNotFoundError(f"DeepSORT模型文件不存在: {cfg.DEEPSORT.REID_CKPT}")
        
        deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )
        logging.info("DeepSORT初始化成功")
        return deepsort
    except Exception as e:
        logging.error(f"DeepSORT初始化失败: {str(e)}")
        raise

# 处理帧
def process_frame(frame, model, deepsort, app):
    try:
        results = model(frame)
        detections = []
        confs = []
        class_ids = []
        counted_ids = set()
        count_person = 0
        count_car = 0
        
        for result in results:
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = box[:4]
                conf = result.boxes.conf.cpu().numpy()[i]
                class_id = int(result.boxes.cls.cpu().numpy()[i])  # 获取类别 ID
                if class_id in CLASS_NAMES:
                    detections.append((x1, y1, x2, y2))
                    confs.append(conf)
                    class_ids.append(class_id)
        
        # Deep SORT 进行目标跟踪
        outputs = []
        if len(detections) > 0:
            bbox_xywh = np.array([[int((x1+x2)/2), int((y1+y2)/2), int(x2-x1), int(y2-y1)] for x1, y1, x2, y2 in detections], dtype=np.float64)
            confs = np.array(confs, dtype=np.float64)
            oids = np.array(class_ids, dtype=np.int32)  # 设定类别 ID
            outputs = deepsort.update(bbox_xywh, confs, oids=oids, ori_img=frame)
        
        for track in outputs:
            if len(track) < 6:
                continue  # 避免崩溃
            x1, y1, x2, y2, track_id, class_id = track
            
            class_name = CLASS_NAMES.get(class_id, "unknown")
            color = CLASS_COLORS.get(class_id, (0, 255, 255))  # 默认黄色
            
            if class_id in CLASS_NAMES:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID: {track_id}, Class: {class_name}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    if class_id == 0:
                        count_person += 1
                    elif class_id == 2:
                        count_car += 1
        
        app.update_counts(count_person, count_car)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        logging.error(f"帧处理失败: {str(e)}")
        return frame

# UI 界面
class App(QWidget):
    def __init__(self):
        try:
            super().__init__()
            self.model_path = r"D:\graduation project\ultralytics-main\best.pt"  # 设置默认模型路径
            self.count_person = 0
            self.count_car = 0
            self.initUI()
            logging.info("UI初始化成功")
        except Exception as e:
            logging.error(f"UI初始化失败: {str(e)}")
            raise

    def initUI(self):
        try:
            self.setWindowTitle("Object Detection UI")
            self.setGeometry(100, 100, 800, 600)
            layout = QVBoxLayout()
            
            self.label = QLabel("选择视频进行检测")
            layout.addWidget(self.label)
            
            self.model_label = QLabel(f"当前模型: {os.path.basename(self.model_path)}")
            layout.addWidget(self.model_label)
            
            self.count_label = QLabel(f"行人数: {self.count_person}，车辆数: {self.count_car}")
            layout.addWidget(self.count_label)
            
            self.btn_select = QPushButton("选择视频")
            self.btn_select.clicked.connect(self.openFile)
            layout.addWidget(self.btn_select)
            
            self.btn_select_model = QPushButton("选择本地模型")
            self.btn_select_model.clicked.connect(self.selectModel)
            layout.addWidget(self.btn_select_model)
            
            self.setLayout(layout)
        except Exception as e:
            logging.error(f"UI布局初始化失败: {str(e)}")
            raise
    
    def update_counts(self, persons, cars):
        try:
            self.count_person = persons
            self.count_car = cars
            self.count_label.setText(f"行人数: {self.count_person}，车辆数: {self.count_car}")
        except Exception as e:
            logging.error(f"更新计数失败: {str(e)}")

    def selectModel(self):
        try:
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO 模型 (*.pt)", options=options)
            if filePath:
                self.model_path = filePath  # 存储用户选择的模型路径
                self.model_label.setText(f"当前模型: {os.path.basename(self.model_path)}")
                logging.info(f"选择模型: {self.model_path}")
        except Exception as e:
            logging.error(f"选择模型失败: {str(e)}")
    
    def openFile(self):
        try:
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi)", options=options)
            if filePath:
                logging.info(f"选择视频: {filePath}")
                self.runDetection(filePath)
        except Exception as e:
            logging.error(f"打开文件失败: {str(e)}")
    
    def runDetection(self, video_path):
        try:
            logging.info("开始视频检测")
            model = load_yolo(self.model_path, 'cuda' if torch.cuda.is_available() else 'cpu')
            deepsort = init_deepsort()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"无法打开视频文件: {video_path}")
                return
                
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                logging.info(f"处理第 {frame_count} 帧")
                
                frame = process_frame(frame, model, deepsort, self)
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                qt_img = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(qt_img))
                QApplication.processEvents()
            
            cap.release()
            cv2.destroyAllWindows()
            logging.info("视频检测完成")
        except Exception as e:
            logging.error(f"运行检测失败: {str(e)}")

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        ex = App()
        ex.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"程序运行失败: {str(e)}")
        raise