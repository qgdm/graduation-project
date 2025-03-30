import torch
import cv2
import numpy as np
import os
import sys

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, 
                           QFileDialog, QVBoxLayout, QHBoxLayout, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter
from PyQt5.QtChart import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics import YOLO

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

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
now_count_person=0
now_count_car=0
count_person = 0
count_car = 0
counted_ids = set()  # 记录已经计数的 ID，防止重复计数

# 初始化YOLO模型
def load_yolo(model_path, device):
    model = YOLO(model_path, task="detect")  # 显式指定任务
    return model

# 初始化Deep SORT
def init_deepsort():
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
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
    return deepsort

# 处理帧
def process_frame(frame, model, deepsort):
    global count_person, count_car, counted_ids ,now_count_person, now_count_car
    results = model(frame)
    detections = []
    confs = []
    class_ids = []

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
        bbox_xywh = np.array(
            [[int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in detections],
            dtype=np.float64)
        confs = np.array(confs, dtype=np.float64)
        oids = np.array(class_ids, dtype=np.int32)  # 设定类别 ID
        outputs = deepsort.update(bbox_xywh, confs, oids=oids, ori_img=frame)

    if outputs is None:
        return frame, count_person, count_car  # 返回更新后的计数

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
                    now_count_person += 1
                elif class_id == 2:
                    count_car += 1
                    now_count_car += 1

    counter_text = f"Current number of people: {now_count_person}, Current number of vehicles: {now_count_car}"
    cv2.putText(frame, counter_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

class App(QWidget):
    def __init__(self):
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
            QChartView {
                background-color: white;
                border-radius: 5px;
            }
        """)

    def initUI(self):
        self.setWindowTitle("智能目标检测系统")
        self.setGeometry(100, 100, 1400, 800)
        
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

        # 创建水平布局来放置视频和图表
        content_layout = QHBoxLayout()
        
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
        self.label.setMinimumSize(800, 450)
        video_layout.addWidget(self.label)
        video_frame.setLayout(video_layout)
        content_layout.addWidget(video_frame)

        # 图表显示区域
        chart_frame = QFrame()
        chart_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
            }
        """)
        chart_layout = QVBoxLayout()
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("目标检测统计")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 创建柱状图系列
        self.series = QBarSeries()
        self.chart.addSeries(self.series)
        
        # 创建数据集
        self.person_set = QBarSet("行人")
        self.car_set = QBarSet("车辆")
        self.person_set.append(0)
        self.car_set.append(0)
        
        # 设置颜色
        self.person_set.setColor(QColor(0, 255, 0))
        self.car_set.setColor(QColor(255, 0, 0))
        
        # 添加数据集到系列
        self.series.append(self.person_set)
        self.series.append(self.car_set)
        
        # 创建坐标轴
        self.axis_x = QBarCategoryAxis()
        self.axis_x.append("当前数量")
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.series.attachAxis(self.axis_x)
        
        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 100)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.series.attachAxis(self.axis_y)
        
        # 创建图表视图
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setMinimumSize(400, 450)
        chart_layout.addWidget(self.chart_view)
        chart_frame.setLayout(chart_layout)
        
        content_layout.addWidget(chart_frame)
        main_layout.addLayout(content_layout)

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

    def updateChart(self):
        """更新图表数据"""
        # 清除现有图表
        self.ax.clear()
        
        # 创建新的柱状图
        self.bars = self.ax.bar(['行人', '车辆'], [now_count_person, now_count_car], color=['green', 'red'])
        self.ax.set_title('目标检测统计')
        
        # 动态调整Y轴范围
        max_value = max(now_count_person, now_count_car)
        self.ax.set_ylim(0, max_value + 10)
        
        # 更新画布
        self.canvas.draw()

    def confirmSelection(self):
        """
        点击确认后，启动图像识别
        """
        if not self.model_path or not self.video_path:
            return  # 如果没有选择模型或视频，不做任何操作

        # 开始图像识别
        self.runDetection(self.video_path)

        # 禁用选择和确认按钮
        self.btn_select.setEnabled(True)
        self.btn_select_model.setEnabled(True)
        self.btn_confirm.setEnabled(False)

    def checkReadyForConfirmation(self):
        """
        检查视频和模型是否都已选择，若已选择，则启用确认按钮
        """
        if self.model_path and self.video_path:
            self.btn_confirm.setEnabled(True)  # 启用确认按钮
        else:
            self.btn_confirm.setEnabled(False)  # 如果还没有选择模型或视频，则禁用确认按钮

    def selectModel(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO 模型 (*.pt)", options=options)
        if filePath:
            self.model_path = filePath  # 存储用户选择的模型路径
            self.model_label.setText(f"当前模型: {os.path.basename(self.model_path)}")

    def openFile(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi)", options=options)
        if filePath:
            self.video_path = filePath  # 存储视频路径
            self.checkReadyForConfirmation()  #更新视频路径，不做图像识别
            self.displayFirstFrame()  # 显示视频的第一帧

    def displayFirstFrame(self):
        """
        显示视频的第一帧作为预览
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if ret:
            # 将视频的第一帧显示在界面上
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            qt_img = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_img))
        cap.release()

    def runDetection(self, video_path):
        global now_count_car,now_count_person,counted_ids
        now_count_car = 0
        now_count_person=0
        counted_ids = set()
        model = load_yolo(self.model_path, 'cuda' if torch.cuda.is_available() else 'cpu')
        deepsort = init_deepsort()
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, model, deepsort)
            # 更新计数标签
            self.counter_label.setText(f"总行人数: {count_person}，总车辆数: {count_car}")
            # 更新图表
            self.updateChart()
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            qt_img = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_img))
            QApplication.processEvents()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_()) 