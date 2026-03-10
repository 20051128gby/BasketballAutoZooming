import json
import cv2
from utils.plots import plot_one_box
from detector import Get_Pred
from ultralytics import YOLO
import myUtils

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        print(f"error: {error}, integral: {self.integral}, derivative: {derivative}, output: {output}")
        return output

class VideoHandler:
    def __init__(self, video_path, start_time=0):
        self.capture = cv2.VideoCapture(video_path)
        self.start_time = start_time  # 开始时间，单位：毫秒
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = self.init_video_writer()

        if not self.set_start_time():
            print("Warning: Unable to set the start time. Starting from the beginning.")

    def init_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20
        return cv2.VideoWriter('output.avi', fourcc, fps, (1000, 600), True)

    def set_start_time(self):
        """设置视频开始时间"""
        if self.start_time > 0:
            return self.capture.set(cv2.CAP_PROP_POS_MSEC, self.start_time)
        return True

    def read_frame(self):
        return self.capture.read()

    def write_frame(self, img, x, y):
        self.video_writer.write(img[y-300:y+300, x-500:x+500])

class ObjectDetector:
    def __init__(self, model_path, ball_model_path):
        self.model = Get_Pred(model_path)
        self.ball_model = YOLO(ball_model_path)
        self.class_names = []
        self.confidence_scores = []

    def detect_objects(self, img):
        value_json = self.model.process(img)
        return json.loads(value_json)['xyxy']

    def detect_ball(self, img, conf=0.4):
        return self.ball_model(img, verbose=False, conf=conf)

    def get_class_and_confidence(self, final_list, frame_count):
        self.class_names, self.confidence_scores = myUtils.print_confidence_score(final_list, frame_count)

class PlayerTracking:
    def __init__(self):
        self.player_center_list = []

    def update_player_centers(self, player_list):
        centers = []
        for bbox in player_list:
            x1, y1, x2, y2 = bbox[:4]
            px_center = (x1 + x2) / 2
            py_center = (y1 + y2) / 2
            centers.append((px_center, py_center))
        return centers

class BasketballAutoZoom:
    def __init__(self, video_path, model_path, ball_model_path, start_time=0):
        self.video_handler = VideoHandler(video_path, start_time=start_time)
        self.detector = ObjectDetector(model_path, ball_model_path)
        self.tracker = PlayerTracking()

        # 初始化PID控制器
        self.pid_x = PIDController(kp=0.2, ki=0.01, kd=0.1)
        self.pid_y = PIDController(kp=0.2, ki=0.01, kd=0.1)

        self.x, self.y = 0, 0  # 初始摄像头中心位置
        self.first_frame = True  # 用于判断是否是第一帧
        self.frame_count = 0  # 用于计数帧数
        self.last_target_x, self.last_target_y = 0, 0  # 上一次的目标位置
        self.min_movement_threshold = 350  # 设定过滤变化过小的中心点变化的阈值

        # 用于平滑处理目标位置
        self.target_position_list = []  # 缓存最近10帧的目标位置
        self.smooth_window = 10  # 设置平滑窗口大小

    def process_video(self):
        while True:
            ret, img = self.video_handler.read_frame()
            if not ret:
                break

            # 每40帧更新一次目标摄像头位置
            if self.frame_count%5==0:
                pred_xyxy = self.detector.detect_objects(img)
                crop_list = myUtils.crop(img, pred_xyxy)
                final_list = myUtils.resize_with_white_background(crop_list)
                self.detector.get_class_and_confidence(final_list, self.frame_count)

                player_list = [
                    pred_xyxy[i]
                    for i in range(len(self.detector.class_names))
                    if self.detector.class_names[i] == 1 and self.detector.confidence_scores[i] > 0.95
                ]
                without_outlier = myUtils.remove_outliers(player_list)
                for i in range(len(without_outlier)):
                    plot_one_box(without_outlier[i], img, label=None, color=(255, 255, 0), line_thickness=1)

                player_centers = self.tracker.update_player_centers(player_list)
                target_x, target_y = myUtils.zoom2(player_centers)

                # 初始化摄像头中心位置
                if self.first_frame:
                    self.x, self.y = target_x, target_y
                    self.first_frame = False

                # 过滤掉变化过小的目标位置
                if abs(target_x - self.last_target_x) < self.min_movement_threshold and abs(target_y - self.last_target_y) < self.min_movement_threshold:
                    target_x, target_y = self.last_target_x, self.last_target_y

                # 更新上次的目标位置
                self.last_target_x, self.last_target_y = target_x, target_y

                
                self.target_position_list.append((target_x, target_y))
                if len(self.target_position_list) > self.smooth_window:
                    self.target_position_list.pop(0)  

                # 计算最近10帧目标位置的平均值（平滑处理）
                smooth_target_x = sum([pos[0] for pos in self.target_position_list]) / len(self.target_position_list)
                smooth_target_y = sum([pos[1] for pos in self.target_position_list]) / len(self.target_position_list)

                # 使用PID控制更新摄像头中心
            self.x -= self.pid_x.update(smooth_target_x - self.x)
            self.y -= self.pid_y.update(smooth_target_y - self.y)

            # 限制摄像头的范围
            self.x = max(500, min(self.video_handler.frame_width - 500, self.x))
            self.y = max(300, min(self.video_handler.frame_height - 300, self.y))

            plot_one_box((self.x - 500, self.y - 300, self.x + 500, self.y + 300), img, label=None, color=(255, 255, 0), line_thickness=1)
            self.video_handler.write_frame(img, int(self.x), int(self.y))

            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_handler.video_writer.release()

# Instantiate and run
if __name__ == "__main__":
    basketball_auto_zoom = BasketballAutoZoom(
        video_path=r"E:\vscodeP\AI\Before.mp4",
        model_path=r'E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\weights\yolov5s.pt',
        ball_model_path=r"E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\basketballModel.pt",
        start_time=6000  # 设置开始时间（单位：毫秒），如60000代表1分钟
    )
    basketball_auto_zoom.process_video()
