import torch
import cv2
import numpy as np
import os
import sys
import logging

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, 
                           QFileDialog, QVBoxLayout, QHBoxLayout, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
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
    2: (255, 0, 0)  # 蓝色：车辆
}

# 计数器
now_count_person = 0
now_count_car = 0
count_person = 0
count_car = 0
counted_ids = set()  # 记录已经计数的 ID，防止重复计数
current_frame_ids = set()  # 记录当前帧中的ID

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
        
        # 修改配置中的类型设置
        cfg.DEEPSORT.MAX_DIST = float(cfg.DEEPSORT.MAX_DIST)
        cfg.DEEPSORT.MIN_CONFIDENCE = float(cfg.DEEPSORT.MIN_CONFIDENCE)
        cfg.DEEPSORT.NMS_MAX_OVERLAP = float(cfg.DEEPSORT.NMS_MAX_OVERLAP)
        cfg.DEEPSORT.MAX_IOU_DISTANCE = float(cfg.DEEPSORT.MAX_IOU_DISTANCE)
        cfg.DEEPSORT.MAX_AGE = int(cfg.DEEPSORT.MAX_AGE)
        cfg.DEEPSORT.N_INIT = int(cfg.DEEPSORT.N_INIT)
        cfg.DEEPSORT.NN_BUDGET = int(cfg.DEEPSORT.NN_BUDGET)
        
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
def process_frame(frame, model, deepsort):
    try:
        global count_person, count_car, counted_ids, now_count_person, now_count_car, current_frame_ids
        current_frame_ids.clear()  # 清空当前帧的ID集合
        
        # 执行YOLO检测
        results = model(frame)
        detections = []
        confs = []
        class_ids = []

        # 打印YOLO检测结果
        logging.info(f"YOLO检测结果数量: {len(results)}")
        
        for result in results:
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = box[:4]
                conf = result.boxes.conf.cpu().numpy()[i]
                class_id = int(result.boxes.cls.cpu().numpy()[i])  # 获取类别 ID
                
                logging.info(f"检测到目标 - 类别: {class_id}, 置信度: {conf:.2f}")
                
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
            oids = np.array(class_ids, dtype=np.int32)  # 使用np.int32
            outputs = deepsort.update(bbox_xywh, confs, oids=oids, ori_img=frame)
            
            logging.info(f"DeepSORT跟踪结果数量: {len(outputs) if outputs is not None else 0}")

        if outputs is None:
            return frame

        # 重置当前帧的计数
        now_count_person = 0
        now_count_car = 0

        for track in outputs:
            if len(track) < 6:
                continue  # 避免崩溃
            x1, y1, x2, y2, track_id, class_id = track

            class_name = CLASS_NAMES.get(class_id, "unknown")
            color = CLASS_COLORS.get(class_id, (0, 255, 255))  # 默认黄色

            if class_id in CLASS_NAMES:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID: {int(track_id)}, Class: {class_name}"  # 确保track_id是整数
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 更新当前帧的计数
                current_frame_ids.add(int(track_id))  # 确保track_id是整数
                if class_id == 0:
                    now_count_person += 1
                    logging.info(f"当前帧行人计数: {now_count_person}")
                elif class_id == 2:
                    now_count_car += 1
                    logging.info(f"当前帧车辆计数: {now_count_car}")

                # 更新总计数（仅对新的ID进行计数）
                if track_id not in counted_ids:
                    counted_ids.add(int(track_id))  # 确保track_id是整数
                    if class_id == 0:
                        count_person += 1
                        logging.info(f"总行人计数: {count_person}")
                    elif class_id == 2:
                        count_car += 1
                        logging.info(f"总车辆计数: {count_car}")

        # 更新计数标签
        counter_text = f"当前帧 - 行人: {now_count_person}, 车辆: {now_count_car}\n总计 - 行人: {count_person}, 车辆: {count_car}"
        cv2.putText(frame, counter_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        logging.error(f"帧处理失败: {str(e)}")
        return frame

class App(QWidget):
    def __init__(self):
        try:
            super().__init__()
            self.model_path = r"D:\graduation project\ultralytics-main\best.pt"
            self.initUI()
            self.setStyleSheet("""
                QWidget {
                    background-color: #f0f0f0;
                    border-radius: 10px;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
                QLabel {
                    color: #333333;
                    font-size: 14px;
                }
                QFrame {
                    background-color: white;
                    border-radius: 5px;
                    padding: 10px;
                }
            """)
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
            
            # 计数信息
            counter_info = QFrame()
            counter_layout = QVBoxLayout()
            self.counter_label = QLabel(f"总行人数: {count_person}，总车辆数: {count_car}")
            counter_layout.addWidget(self.counter_label)
            counter_info.setLayout(counter_layout)
            
            info_layout.addWidget(model_info)
            info_layout.addWidget(counter_info)
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
            
            self.btn_confirm = QPushButton("开始检测")
            self.btn_confirm.setEnabled(False)
            self.btn_confirm.clicked.connect(self.confirmSelection)
            
            button_layout.addWidget(self.btn_select)
            button_layout.addWidget(self.btn_select_model)
            button_layout.addWidget(self.btn_confirm)
            button_frame.setLayout(button_layout)
            main_layout.addWidget(button_frame)

            self.setLayout(main_layout)
            logging.info("UI布局初始化成功")
        except Exception as e:
            logging.error(f"UI布局初始化失败: {str(e)}")
            raise

    def confirmSelection(self):
        """
        点击确认后，启动图像识别
        """
        try:
            if not self.model_path or not self.video_path:
                return  # 如果没有选择模型或视频，不做任何操作

            # 开始图像识别
            self.runDetection(self.video_path)

            # 禁用选择和确认按钮
            self.btn_select.setEnabled(True)
            self.btn_select_model.setEnabled(True)
            self.btn_confirm.setEnabled(False)
        except Exception as e:
            logging.error(f"确认选择失败: {str(e)}")

    def checkReadyForConfirmation(self):
        """
        检查视频和模型是否都已选择，若已选择，则启用确认按钮
        """
        try:
            if self.model_path and self.video_path:
                self.btn_confirm.setEnabled(True)  # 启用确认按钮
            else:
                self.btn_confirm.setEnabled(False)  # 如果还没有选择模型或视频，则禁用确认按钮
        except Exception as e:
            logging.error(f"检查确认状态失败: {str(e)}")

    def selectModel(self):
        try:
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO 模型 (*.pt)", options=options)
            if filePath:
                self.model_path = filePath  # 存储用户选择的模型路径
                self.model_label.setText(f"当前模型: {os.path.basename(self.model_path)}")
        except Exception as e:
            logging.error(f"选择模型失败: {str(e)}")

    def openFile(self):
        try:
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi)", options=options)
            if filePath:
                self.video_path = filePath  # 存储视频路径
                self.checkReadyForConfirmation()  #更新视频路径，不做图像识别
                self.displayFirstFrame()  # 显示视频的第一帧
        except Exception as e:
            logging.error(f"打开文件失败: {str(e)}")

    def displayFirstFrame(self):
        """
        显示视频的第一帧作为预览
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                # 将视频的第一帧显示在界面上
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                qt_img = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(qt_img))
            cap.release()
        except Exception as e:
            logging.error(f"显示第一帧失败: {str(e)}")

    def runDetection(self, video_path):
        try:
            global now_count_car, now_count_person, counted_ids, current_frame_ids, count_person, count_car
            # 重置所有计数器
            now_count_car = 0
            now_count_person = 0
            count_person = 0
            count_car = 0
            counted_ids.clear()
            current_frame_ids.clear()
            
            logging.info("开始视频检测")
            model = load_yolo(self.model_path, 'cuda' if torch.cuda.is_available() else 'cpu')
            deepsort = init_deepsort()
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                logging.info(f"处理第 {frame_count} 帧")
                
                frame = process_frame(frame, model, deepsort)
                # 更新计数标签
                self.counter_label.setText(f"总行人数: {count_person}，总车辆数: {count_car}")
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