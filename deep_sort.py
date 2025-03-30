import torch
import cv2
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
count_person = 0
count_car = 0
counted_ids = set()  # 记录已经计数的 ID，防止重复计数


# 初始化YOLOv11模型
def load_yolo(model_path, device):
    model = YOLO(model_path)  # 加载模型
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
    global count_person, count_car, counted_ids
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
    if len(detections) > 0:
        bbox_xywh = np.array(
            [[int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in detections],
            dtype=np.float64)
        confs = np.array(confs, dtype=np.float64)
        oids = np.array(class_ids, dtype=np.int32)  # 设定类别 ID
        outputs = deepsort.update(bbox_xywh, confs, oids=oids, ori_img=frame)
    else:
        outputs = []

    for track in outputs:
        if len(track) == 6:
            x1, y1, x2, y2, track_id, class_id = track
        elif len(track) == 5:
            x1, y1, x2, y2, track_id = track
            class_id = -1  # 默认类别 ID
        else:
            print("Unexpected output format:", track)
            continue  # 跳过错误数据

        class_name = CLASS_NAMES.get(class_id, "unknown")
        color = CLASS_COLORS.get(class_id, (0, 255, 255))  # 默认黄色

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"ID: {track_id}, Class: {class_name}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if track_id not in counted_ids:
            counted_ids.add(track_id)
            if class_id == 0:
                count_person += 1
            elif class_id == 2:
                count_car += 1

    counter_text = f"Persons: {count_person}, Cars: {count_car}"
    cv2.putText(frame, counter_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame


# 主函数
def main(video_path, model_path):
    model = load_yolo(model_path, 'cuda' if torch.cuda.is_available() else 'cpu')
    deepsort = init_deepsort()

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, deepsort)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(r'D:\graduation project\ultralytics-main\datasets\Argoverse\images\val1\ring_front_center_315966774109236128.jpg', r'D:\graduation project\ultralytics-main\best.pt')
