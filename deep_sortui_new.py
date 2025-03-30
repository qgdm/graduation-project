import torch
import cv2
import numpy as np
import os
import sys
import logging

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, 
                           QFileDialog, QVBoxLayout, QHBoxLayout, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics import YOLO

if not hasattr(np, 'int'):
    np.int = int  # 兼容 NumPy 1.20+ 移除 np.int 的问题

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
    2: (255, 0, 0)  # 蓝色：车辆
}

class App(QWidget):
    def __init__(self):
        try:
            super().__init__()
            self.model_path = r"D:\graduation project\ultralytics-main\best.pt"
            self.video_path = None
            self.is_running = False
            self.is_paused = False
            self.count_person = 0  # 当前视频计数
            self.count_car = 0     # 当前视频计数
            self.total_count_person = 0  # 所有视频总计数
            self.total_count_car = 0     # 所有视频总计数
            self.counted_ids = set()
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_frame)
            self.initUI()
            logging.info("UI初始化成功")
        except Exception as e:
            logging.error(f"UI初始化失败: {str(e)}")
            raise

    def initUI(self):
        try:
            self.setWindowTitle("智能目标检测系统")
            self.setGeometry(100, 100, 1200, 800)
            
            # 主布局
            main_layout = QVBoxLayout()
            main_layout.setSpacing(20)
            main_layout.setContentsMargins(20, 20, 20, 20)

            # 标题
            title_label = QLabel("智能目标检测系统")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(QFont('Arial', 24, QFont.Bold))
            title_label.setStyleSheet("color: #2c3e50; margin: 20px;")
            main_layout.addWidget(title_label)

            # 视频显示区域
            video_frame = QFrame()
            video_frame.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border: 2px solid #bdc3c7;
                    border-radius: 10px;
                }
            """)
            video_layout = QVBoxLayout()
            self.label = QLabel("选择视频进行检测")
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setMinimumSize(1000, 600)
            video_layout.addWidget(self.label)
            video_frame.setLayout(video_layout)
            main_layout.addWidget(video_frame)

            # 信息显示区域
            info_frame = QFrame()
            info_layout = QHBoxLayout()
            
            # 模型信息
            model_info = QFrame()
            model_layout = QVBoxLayout()
            self.model_label = QLabel(f"当前模型: {os.path.basename(self.model_path)}")
            model_layout.addWidget(self.model_label)
            model_info.setLayout(model_layout)
            
            # 当前视频计数信息
            current_counter_info = QFrame()
            current_counter_layout = QVBoxLayout()
            self.current_counter_label = QLabel(f"当前视频 - 行人: {self.count_person}, 车辆: {self.count_car}")
            current_counter_layout.addWidget(self.current_counter_label)
            current_counter_info.setLayout(current_counter_layout)
            
            # 总计数信息
            total_counter_info = QFrame()
            total_counter_layout = QVBoxLayout()
            self.total_counter_label = QLabel(f"总计 - 行人: {self.total_count_person}, 车辆: {self.total_count_car}")
            total_counter_layout.addWidget(self.total_counter_label)
            total_counter_info.setLayout(total_counter_layout)
            
            info_layout.addWidget(model_info)
            info_layout.addWidget(current_counter_info)
            info_layout.addWidget(total_counter_info)
            info_frame.setLayout(info_layout)
            main_layout.addWidget(info_frame)

            # 按钮区域
            button_frame = QFrame()
            button_layout = QHBoxLayout()
            button_layout.setSpacing(20)
            
            self.btn_select = QPushButton("选择视频")
            self.btn_select.clicked.connect(self.openFile)
            
            self.btn_select_model = QPushButton("选择模型")
            self.btn_select_model.clicked.connect(self.selectModel)
            
            self.btn_start = QPushButton("开始检测")
            self.btn_start.setEnabled(False)
            self.btn_start.clicked.connect(self.startDetection)
            
            self.btn_pause = QPushButton("暂停")
            self.btn_pause.setEnabled(False)
            self.btn_pause.clicked.connect(self.pauseDetection)
            
            button_layout.addWidget(self.btn_select)
            button_layout.addWidget(self.btn_select_model)
            button_layout.addWidget(self.btn_start)
            button_layout.addWidget(self.btn_pause)
            button_frame.setLayout(button_layout)
            main_layout.addWidget(button_frame)

            self.setLayout(main_layout)
            logging.info("UI布局初始化成功")
        except Exception as e:
            logging.error(f"UI布局初始化失败: {str(e)}")
            raise

    def initDetection(self):
        try:
            self.model = YOLO(self.model_path, task="detect")
            logging.info(f"YOLO模型加载成功: {self.model_path}")
            
            cfg = get_config()
            cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
            cfg.DEEPSORT.REID_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                "deep_sort_pytorch", "deep_sort", "deep", "checkpoint", "ckpt.t7")
            
            if not os.path.exists(cfg.DEEPSORT.REID_CKPT):
                logging.error(f"DeepSORT模型文件不存在: {cfg.DEEPSORT.REID_CKPT}")
                raise FileNotFoundError(f"DeepSORT模型文件不存在: {cfg.DEEPSORT.REID_CKPT}")
            
            self.deepsort = DeepSort(
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
            
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logging.error(f"无法打开视频文件: {self.video_path}")
                raise FileNotFoundError(f"无法打开视频文件: {self.video_path}")
            
            return True
        except Exception as e:
            logging.error(f"初始化检测失败: {str(e)}")
            return False

    def process_frame(self):
        try:
            if not self.is_running or self.is_paused:
                return
                
            ret, frame = self.cap.read()
            if not ret:
                self.stopDetection()
                return
            
            # 执行YOLO检测
            results = self.model(frame)
            detections = []
            confs = []
            class_ids = []
            
            for result in results:
                for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box[:4]
                    conf = result.boxes.conf.cpu().numpy()[i]
                    class_id = int(result.boxes.cls.cpu().numpy()[i])
                    if class_id in CLASS_NAMES:
                        detections.append((x1, y1, x2, y2))
                        confs.append(conf)
                        class_ids.append(class_id)
            
            # Deep SORT 进行目标跟踪
            outputs = []
            if len(detections) > 0:
                bbox_xywh = np.array(
                    [[int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in detections],
                    dtype=np.float64)
                confs = np.array(confs, dtype=np.float64)
                oids = np.array(class_ids, dtype=np.int32)
                outputs = self.deepsort.update(bbox_xywh, confs, oids=oids, ori_img=frame)
            
            if outputs is not None:
                for track in outputs:
                    if len(track) < 6:
                        continue
                    x1, y1, x2, y2, track_id, class_id = track
                    
                    class_name = CLASS_NAMES.get(class_id, "unknown")
                    color = CLASS_COLORS.get(class_id, (0, 255, 255))
                    
                    if class_id in CLASS_NAMES:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"ID: {int(track_id)}, Class: {class_name}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 更新计数
                        if track_id not in self.counted_ids:
                            self.counted_ids.add(int(track_id))
                            if class_id == 0:
                                self.count_person += 1
                                self.total_count_person += 1
                            elif class_id == 2:
                                self.count_car += 1
                                self.total_count_car += 1
            
            # 更新计数标签
            self.current_counter_label.setText(f"当前视频 - 行人: {self.count_person}, 车辆: {self.count_car}")
            self.total_counter_label.setText(f"总计 - 行人: {self.total_count_person}, 车辆: {self.total_count_car}")
            
            # 显示帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            qt_img = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_img))
            
        except Exception as e:
            logging.error(f"帧处理失败: {str(e)}")

    def startDetection(self):
        try:
            if not self.video_path:
                return
                
            if not self.is_running:
                if self.initDetection():
                    self.is_running = True
                    self.is_paused = False
                    self.timer.start(30)  # 约33fps
                    self.btn_start.setText("停止检测")
                    self.btn_pause.setEnabled(True)
                    logging.info("开始检测")
        except Exception as e:
            logging.error(f"启动检测失败: {str(e)}")

    def pauseDetection(self):
        try:
            self.is_paused = not self.is_paused
            self.btn_pause.setText("继续" if self.is_paused else "暂停")
            logging.info(f"检测{'暂停' if self.is_paused else '继续'}")
        except Exception as e:
            logging.error(f"暂停检测失败: {str(e)}")

    def stopDetection(self):
        try:
            self.is_running = False
            self.is_paused = False
            self.timer.stop()
            if hasattr(self, 'cap'):
                self.cap.release()
            self.btn_start.setText("开始检测")
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("暂停")
            logging.info("停止检测")
        except Exception as e:
            logging.error(f"停止检测失败: {str(e)}")

    def selectModel(self):
        try:
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO 模型 (*.pt)", options=options)
            if filePath:
                self.model_path = filePath
                self.model_label.setText(f"当前模型: {os.path.basename(self.model_path)}")
                logging.info(f"选择模型: {self.model_path}")
        except Exception as e:
            logging.error(f"选择模型失败: {str(e)}")

    def openFile(self):
        try:
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi)", options=options)
            if filePath:
                self.video_path = filePath
                self.stopDetection()
                # 重置当前视频的计数器和ID集合
                self.count_person = 0
                self.count_car = 0
                self.counted_ids.clear()
                self.displayFirstFrame()
                self.btn_start.setEnabled(True)
                logging.info(f"选择视频: {self.video_path}")
        except Exception as e:
            logging.error(f"打开文件失败: {str(e)}")

    def displayFirstFrame(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                qt_img = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(qt_img))
            cap.release()
        except Exception as e:
            logging.error(f"显示第一帧失败: {str(e)}")

    def closeEvent(self, event):
        self.stopDetection()
        event.accept()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        ex = App()
        ex.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"程序运行失败: {str(e)}")
        raise 