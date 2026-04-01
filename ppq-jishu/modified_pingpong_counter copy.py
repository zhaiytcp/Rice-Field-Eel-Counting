import sys
import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import cv2
import time
import numpy as np
from collections import defaultdict

import torch
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


# ===================== 全局配置 =====================
MODEL_PATH = r'E:\Visual-project\ultralytics-jishu\ppq-jishu\best.engine'
TRACKER_CFG = "bytetrack1.0.yaml"

CONF_THRESH = 0.01
TRACK_BUFFER = 180
PROCESS_WIDTH = 500
EMIT_EVERY_N_FRAMES = 1
MAX_TRAIL_SEGMENTS = 100

CAMERA_INDEX = 0
RECORD_FPS = 120.0
FRAME_SIZE = (800, 600)

# 为了更稳地“边写边读”，默认改成 AVI + MJPG
# 如果你坚持 MP4，也可以把后缀和 FOURCC 改回去，但边录边读通常没 AVI 稳。
RECORD_FILE_EXT = ".avi"
RECORD_FOURCC = "MJPG"
ANALYZED_FOURCC = "MJPG"

# growing-file 追读相关
POLL_INTERVAL_WHEN_WAITING = 0.05
OPEN_RETRY_INTERVAL = 0.10
EOF_CONFIRM_RETRIES = 8

# 计数逻辑相关
COUNT_MARGIN_PX = 8
COUNT_COOLDOWN_FRAMES = 6
MIN_TRACK_LENGTH_FOR_COUNT = 5
MAX_MISSES_KEEP_TRACK = 10


# ===================== 工具函数 =====================
def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def smooth_positions(positions, window_size=3):
    if not positions:
        return None
    if len(positions) < window_size:
        return positions[-1]
    return np.mean(positions[-window_size:], axis=0)


def side_with_margin(x, y, A, B, C, margin=8):
    denom = max(np.sqrt(A * A + B * B), 1e-6)
    dist = (A * x + B * y + C) / denom
    if dist > margin:
        return 1
    if dist < -margin:
        return -1
    return 0


# ===================== 录制线程 =====================
class RecorderThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, float, int)       # frame, fps, recorded_frames
    recording_started = pyqtSignal(str, int, int, float)   # path, width, height, fps
    recording_finished = pyqtSignal(str, int)              # path, total_frames
    recording_error = pyqtSignal(str)

    def __init__(self, camera_index=0, fps=120.0, frame_size=(800, 600)):
        super().__init__()
        self.camera_index = camera_index
        self.target_fps = fps
        self.frame_size = frame_size
        self.is_recording = False
        self.output_file = ""

    def stop(self):
        self.is_recording = False

    def run(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = f"output_{timestamp}{RECORD_FILE_EXT}"

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        if not cap.isOpened():
            self.recording_error.emit("无法打开摄像头")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = safe_float(cap.get(cv2.CAP_PROP_FPS), self.target_fps)

        if actual_width <= 0 or actual_height <= 0:
            cap.release()
            self.recording_error.emit("无法获取摄像头分辨率")
            return

        if actual_fps <= 0 or not np.isfinite(actual_fps):
            actual_fps = self.target_fps

        fourcc = cv2.VideoWriter_fourcc(*RECORD_FOURCC)
        out = cv2.VideoWriter(self.output_file, fourcc, actual_fps, (actual_width, actual_height))

        if not out.isOpened():
            cap.release()
            self.recording_error.emit("无法创建输出视频文件")
            return

        self.is_recording = True
        self.recording_started.emit(self.output_file, actual_width, actual_height, actual_fps)

        prev_time = time.time()
        frame_count = 0
        current_fps = 0.0
        recorded_frames = 0

        try:
            while self.is_recording:
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)
                recorded_frames += 1

                frame_count += 1
                now = time.time()
                elapsed = now - prev_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    prev_time = now

                show_frame = frame.copy()
                cv2.putText(show_frame, f"Recording FPS: {current_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(show_frame, "Recording...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(show_frame, f"Frames: {recorded_frames}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(show_frame, f"Res: {actual_width}x{actual_height}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

                self.frame_ready.emit(show_frame, current_fps, recorded_frames)

        except Exception as e:
            self.recording_error.emit(f"录制异常: {e}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            try:
                out.release()
            except Exception:
                pass
            self.recording_finished.emit(self.output_file, recorded_frames)


# ===================== 追读视频文件并分析的线程 =====================
class GrowingVideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray, float, int, int, int)  # frame, fps, count, processed_frames, available_frames
    processing_finished = pyqtSignal(int)
    processing_error = pyqtSignal(str)

    def __init__(
        self,
        video_path,
        count_line,
        source_width=None,
        source_height=None,
        source_fps=None,
        is_recording_func=None,
        get_available_frames_func=None,
    ):
        super().__init__()
        self.video_path = video_path
        self.count_line_orig = count_line
        self.source_width = source_width
        self.source_height = source_height
        self.source_fps = source_fps or 25.0
        self.is_recording_func = is_recording_func
        self.get_available_frames_func = get_available_frames_func

        self.is_processing = False
        self.total_count = 0
        self.processed_frames = 0
        self.available_frames = 0
        self.output_path = ""
        self.save_output = True

        self.process_width = PROCESS_WIDTH
        self.process_height = 0

        self.performance_stats = {
            'inference_time': [],
            'tracking_time': [],
            'drawing_time': [],
            'total_frame_time': []
        }

    def stop_processing(self):
        self.is_processing = False

    def _still_recording(self):
        try:
            return bool(self.is_recording_func()) if self.is_recording_func else False
        except Exception:
            return False

    def _get_available_frames(self):
        if self.get_available_frames_func is not None:
            try:
                value = int(self.get_available_frames_func())
                if value >= 0:
                    return value
            except Exception:
                pass

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return 0
        frames = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
        cap.release()
        return max(0, frames)

    def _ensure_video_info(self):
        if self.source_width and self.source_height:
            return True

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False

        self.source_width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0)
        self.source_height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
        if fps > 0 and np.isfinite(fps):
            self.source_fps = fps
        cap.release()

        return self.source_width > 0 and self.source_height > 0

    def _open_capture_at_processed_position(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None

        if self.processed_frames > 0:
            ok = cap.set(cv2.CAP_PROP_POS_FRAMES, self.processed_frames)
            if not ok:
                # 某些后端 set 失败时，后续 read 仍可尝试；这里只是不强依赖返回值。
                pass
        return cap

    def _prepare_output_writer(self):
        if not self.save_output:
            return None

        base, _ext = os.path.splitext(self.video_path)
        self.output_path = base + "_analyzed" + RECORD_FILE_EXT
        fourcc = cv2.VideoWriter_fourcc(*ANALYZED_FOURCC)
        writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            max(1.0, float(self.source_fps)),
            (self.process_width, self.process_height)
        )
        if not writer.isOpened():
            return None
        return writer

    def run(self):
        video_writer = None
        cap = None

        try:
            wait_start = time.time()
            while self.is_processing is False:
                self.is_processing = True

            # 等待视频文件出现并可读
            while self.is_processing:
                if os.path.exists(self.video_path) and self._ensure_video_info():
                    break
                if not self._still_recording() and time.time() - wait_start > 3.0:
                    self.processing_error.emit("无法获取视频信息")
                    self.processing_finished.emit(0)
                    return
                time.sleep(OPEN_RETRY_INTERVAL)

            self.process_height = int(self.source_height * self.process_width / self.source_width)

            sx = self.process_width / self.source_width
            sy = self.process_height / self.source_height
            process_count_line = [
                int(self.count_line_orig[0] * sx),
                int(self.count_line_orig[1] * sy),
                int(self.count_line_orig[2] * sx),
                int(self.count_line_orig[3] * sy)
            ]

            model = YOLO(MODEL_PATH, task="detect")

            warmup = np.zeros((self.process_height, self.process_width, 3), dtype=np.uint8)
            _ = model.predict(
                warmup,
                imgsz=self.process_width,
                conf=CONF_THRESH,
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=False
            )

            video_writer = self._prepare_output_writer()
            self.available_frames = self._get_available_frames()

            fps_counter = 0
            fps_t0 = time.time()
            current_fps = 0.0
            eof_retry_count = 0

            track_history = defaultdict(lambda: {
                "positions": [],
                "prev_side": None,
                "last_seen": 0,
                "consecutive_misses": 0,
                "track_length": 0,
                "last_count_frame": -9999,
                "color": tuple(np.random.randint(0, 255, 3).tolist())
            })

            (lx1, ly1, lx2, ly2) = process_count_line
            A = ly2 - ly1
            B = lx1 - lx2
            C = lx2 * ly1 - lx1 * ly2

            while self.is_processing:
                if cap is None:
                    cap = self._open_capture_at_processed_position()
                    if cap is None:
                        if self._still_recording():
                            time.sleep(OPEN_RETRY_INTERVAL)
                            continue
                        self.available_frames = self._get_available_frames()
                        if self.available_frames > self.processed_frames:
                            time.sleep(OPEN_RETRY_INTERVAL)
                            continue
                        break

                frame_t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None

                    self.available_frames = self._get_available_frames()

                    if self._still_recording():
                        eof_retry_count = 0
                        time.sleep(POLL_INTERVAL_WHEN_WAITING)
                        continue

                    if self.available_frames > self.processed_frames:
                        time.sleep(POLL_INTERVAL_WHEN_WAITING)
                        continue

                    eof_retry_count += 1
                    if eof_retry_count >= EOF_CONFIRM_RETRIES:
                        break
                    time.sleep(POLL_INTERVAL_WHEN_WAITING)
                    continue

                eof_retry_count = 0
                self.processed_frames += 1
                fps_counter += 1
                self.available_frames = max(self.available_frames, self._get_available_frames())

                now = time.time()
                if now - fps_t0 >= 1.0:
                    current_fps = fps_counter / max(now - fps_t0, 1e-6)
                    fps_counter = 0
                    fps_t0 = now

                resize_frame = cv2.resize(frame, (self.process_width, self.process_height))

                inf_t0 = time.perf_counter()
                results = model.track(
                    resize_frame,
                    persist=True,
                    conf=CONF_THRESH,
                    tracker=TRACKER_CFG,
                    verbose=False,
                    imgsz=640,
                    device=0 if torch.cuda.is_available() else 'cpu'
                )
                inf_t1 = time.perf_counter()
                inference_time = inf_t1 - inf_t0
                self.performance_stats['inference_time'].append(inference_time)

                current_objs = {}
                trk_t0 = time.perf_counter()

                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for box, tid, conf in zip(boxes, track_ids, confidences):
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        current_objs[tid] = {
                            'center': (cx, cy),
                            'box': box,
                            'confidence': conf
                        }

                        th = track_history[tid]
                        th["positions"].append((cx, cy))
                        if len(th["positions"]) > TRACK_BUFFER:
                            th["positions"].pop(0)
                        th["last_seen"] = self.processed_frames
                        th["consecutive_misses"] = 0
                        th["track_length"] += 1

                active_tracks = 0
                for tid in list(track_history.keys()):
                    if tid in current_objs:
                        active_tracks += 1
                    else:
                        track_history[tid]["consecutive_misses"] += 1
                        if (
                            track_history[tid]["consecutive_misses"] > MAX_MISSES_KEEP_TRACK
                            or track_history[tid]["track_length"] < 2
                        ):
                            del track_history[tid]
                        else:
                            active_tracks += 1

                for tid, hist in list(track_history.items()):
                    if tid not in current_objs:
                        continue

                    pos = hist["positions"]
                    if len(pos) < 2:
                        continue

                    sp = smooth_positions(pos)
                    if sp is None:
                        continue

                    side = side_with_margin(sp[0], sp[1], A, B, C, margin=COUNT_MARGIN_PX)
                    prev = hist["prev_side"]

                    if (
                        prev == 1 and side == -1
                        and hist["track_length"] >= MIN_TRACK_LENGTH_FOR_COUNT
                        and self.processed_frames - hist["last_count_frame"] > COUNT_COOLDOWN_FRAMES
                    ):
                        self.total_count += 1
                        hist["last_count_frame"] = self.processed_frames

                    if side != 0:
                        hist["prev_side"] = side

                trk_t1 = time.perf_counter()
                self.performance_stats['tracking_time'].append(trk_t1 - trk_t0)

                draw_t0 = time.perf_counter()
                draw_frame = resize_frame.copy()
                cv2.line(draw_frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)

                for tid, hist in track_history.items():
                    if tid not in current_objs:
                        continue

                    color = hist["color"]
                    x1, y1, x2, y2 = current_objs[tid]['box']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(draw_frame, f'ID:{tid}', (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    pts = hist["positions"]
                    start_i = max(1, len(pts) - MAX_TRAIL_SEGMENTS)
                    for i in range(start_i, len(pts)):
                        p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                        p1 = (int(pts[i][0]), int(pts[i][1]))
                        cv2.line(draw_frame, p0, p1, color, 1)

                lag_frames = max(0, self.available_frames - self.processed_frames)
                live_text = "LIVE" if self._still_recording() else "FINALIZING"

                cv2.putText(draw_frame, f"Total Count: {self.total_count}", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(draw_frame, f"FPS: {current_fps:.1f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(draw_frame, f"Infer: {inference_time * 1000:.1f}ms", (20, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(draw_frame, f"Active: {active_tracks}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(draw_frame, f"Processed: {self.processed_frames}", (20, 134),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(draw_frame, f"Available: {self.available_frames}", (20, 158),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(draw_frame, f"Lag: {lag_frames}", (20, 182),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(draw_frame, f"Mode: {live_text}", (20, 206),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                draw_t1 = time.perf_counter()
                self.performance_stats['drawing_time'].append(draw_t1 - draw_t0)
                self.performance_stats['total_frame_time'].append(time.perf_counter() - frame_t0)

                if video_writer is not None:
                    video_writer.write(draw_frame)

                if (self.processed_frames % EMIT_EVERY_N_FRAMES) == 0:
                    self.frame_processed.emit(
                        draw_frame,
                        current_fps,
                        self.total_count,
                        self.processed_frames,
                        self.available_frames
                    )

        except Exception as e:
            self.processing_error.emit(f"分析异常: {e}")
        finally:
            self.is_processing = False
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                if video_writer is not None:
                    video_writer.release()
            except Exception:
                pass
            self.processing_finished.emit(self.total_count)


# ===================== 主界面 =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recorder = None
        self.processor = None

        self.video_path = ""
        self.count_line = [300, 2, 300, 600]

        self.current_frame = None
        self.current_fps = 0.0
        self.current_count = 0
        self.current_processed_frames = 0
        self.current_available_frames = 0
        self.current_recorded_frames = 0

        self.source_width = 0
        self.source_height = 0
        self.source_fps = 25.0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("乒乓球录制 + 边录边分析计数")
        self.setGeometry(100, 100, 1280, 860)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("录制路径会显示在这里，也可以手动加载已有视频")
        path_layout.addWidget(self.file_path_edit)

        self.browse_btn = QPushButton("加载已有视频")
        self.browse_btn.clicked.connect(self.browse_video_file)
        path_layout.addWidget(self.browse_btn)

        main_layout.addLayout(path_layout)

        self.video_label = QLabel("画面预览区域")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(900, 600)
        self.video_label.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.video_label)

        info_layout = QHBoxLayout()

        self.mode_label = QLabel("状态: 待机")
        self.mode_label.setStyleSheet("font-weight: bold; color: purple;")
        info_layout.addWidget(self.mode_label)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("font-weight: bold; color: green;")
        info_layout.addWidget(self.fps_label)

        self.count_label = QLabel("总计数: 0")
        self.count_label.setStyleSheet("font-weight: bold; color: blue;")
        info_layout.addWidget(self.count_label)

        self.progress_label = QLabel("处理: 0 / 0")
        self.progress_label.setStyleSheet("font-weight: bold; color: orange;")
        info_layout.addWidget(self.progress_label)

        main_layout.addLayout(info_layout)

        btn_layout = QHBoxLayout()

        self.start_record_btn = QPushButton("开始录制")
        self.start_record_btn.clicked.connect(self.start_recording)
        btn_layout.addWidget(self.start_record_btn)

        self.stop_record_btn = QPushButton("停止录制")
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.stop_record_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_record_btn)

        self.start_analysis_btn = QPushButton("开始分析")
        self.start_analysis_btn.clicked.connect(self.start_processing_loaded_video)
        self.start_analysis_btn.setEnabled(False)
        btn_layout.addWidget(self.start_analysis_btn)

        self.stop_analysis_btn = QPushButton("停止分析")
        self.stop_analysis_btn.clicked.connect(self.stop_processing)
        self.stop_analysis_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_analysis_btn)

        main_layout.addLayout(btn_layout)

        self.status_label = QLabel("点击“开始录制”后，会边录边分析；停止录制后，分析会继续跑到视频末尾")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.status_label)

    # ---------- 公共显示 ----------
    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def preview_video_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                h, w = frame.shape[:2]
                self.source_width = w
                self.source_height = h
                self.source_fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 25.0)
                self.count_line = [w // 2, int(h * 0.01), w // 2, int(h * 0.99)]
            cap.release()

    def reset_stats(self):
        self.current_count = 0
        self.current_processed_frames = 0
        self.current_available_frames = 0
        self.current_recorded_frames = 0
        self.current_fps = 0.0
        self.count_label.setText("总计数: 0")
        self.fps_label.setText("FPS: 0.0")
        self.progress_label.setText("处理: 0 / 0")

    def is_recording_now(self):
        return self.recorder is not None and self.recorder.isRunning()

    def get_current_recorded_frames(self):
        return self.current_recorded_frames

    # ---------- 手动加载视频 ----------
    def browse_video_file(self):
        if self.is_recording_now():
            self.status_label.setText("录制中，不能加载其它视频")
            return

        if self.processor and self.processor.isRunning():
            self.status_label.setText("分析中，不能加载其它视频")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)"
        )
        if file_path:
            self.video_path = file_path
            self.file_path_edit.setText(file_path)
            self.preview_video_first_frame(file_path)
            self.start_analysis_btn.setEnabled(True)
            self.status_label.setText("视频已加载，可以开始分析")
            self.mode_label.setText("状态: 已加载视频")
            self.stop_analysis_btn.setEnabled(False)
            self.start_record_btn.setEnabled(True)

    # ---------- 录制 ----------
    def start_recording(self):
        if self.processor and self.processor.isRunning():
            self.status_label.setText("分析进行中，不能开始录制")
            return

        self.video_path = ""
        self.file_path_edit.clear()
        self.reset_stats()

        self.recorder = RecorderThread(
            camera_index=CAMERA_INDEX,
            fps=RECORD_FPS,
            frame_size=FRAME_SIZE
        )
        self.recorder.frame_ready.connect(self.update_recording_frame)
        self.recorder.recording_started.connect(self.on_recording_started)
        self.recorder.recording_finished.connect(self.on_recording_finished)
        self.recorder.recording_error.connect(self.on_recording_error)

        self.start_record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        self.start_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)

        self.mode_label.setText("状态: 录制中")
        self.status_label.setText("正在录制，等待分析线程接入...")
        self.recorder.start()

    def stop_recording(self):
        if self.recorder and self.recorder.isRunning():
            self.status_label.setText("已发出停止录制指令，分析会继续处理剩余帧...")
            self.recorder.stop()

    def on_recording_started(self, path, width, height, fps):
        self.video_path = path
        self.file_path_edit.setText(path)
        self.source_width = width
        self.source_height = height
        self.source_fps = fps
        self.count_line = [width // 2, int(height * 0.01), width // 2, int(height * 0.99)]
        self.status_label.setText(f"录制开始：{width}x{height} @ {fps:.2f}fps，正在启动分析...")

        # 录制一开始就启动追读分析
        self.start_processing_live_recording()

    def on_recording_finished(self, path, total_frames):
        self.video_path = path
        self.file_path_edit.setText(path)
        self.current_recorded_frames = total_frames

        self.start_record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(False)

        if total_frames <= 0:
            self.mode_label.setText("状态: 待机")
            self.status_label.setText("录制失败或没有录到有效帧")
            if not (self.processor and self.processor.isRunning()):
                self.start_record_btn.setEnabled(True)
                self.browse_btn.setEnabled(True)
            return

        self.mode_label.setText("状态: 录制结束，分析收尾中")
        self.status_label.setText(f"录制完成，共 {total_frames} 帧；分析线程正在继续处理剩余帧...")
        self.stop_analysis_btn.setEnabled(True)

    def on_recording_error(self, msg):
        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.start_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(False)

        self.mode_label.setText("状态: 错误")
        self.status_label.setText(msg)

    def update_recording_frame(self, frame, fps, recorded_frames):
        self.current_recorded_frames = recorded_frames

        # 只在分析线程还没出图前，显示录制预览
        if not (self.processor and self.processor.isRunning() and self.current_processed_frames > 0):
            self.current_frame = frame
            self.display_frame(frame)

        self.fps_label.setText(f"录制FPS: {fps:.1f}")
        self.progress_label.setText(f"处理: {self.current_processed_frames} / 已录制: {recorded_frames}")

    # ---------- 分析 ----------
    def _create_processor(self, live_mode):
        return GrowingVideoProcessor(
            video_path=self.video_path,
            count_line=self.count_line,
            source_width=self.source_width,
            source_height=self.source_height,
            source_fps=self.source_fps,
            is_recording_func=(self.is_recording_now if live_mode else None),
            get_available_frames_func=(self.get_current_recorded_frames if live_mode else None),
        )

    def start_processing_live_recording(self):
        if not self.video_path:
            self.status_label.setText("没有可分析的视频路径")
            return

        if self.processor and self.processor.isRunning():
            return

        self.processor = self._create_processor(live_mode=True)
        self.processor.frame_processed.connect(self.update_processed_frame)
        self.processor.processing_finished.connect(self.processing_finished)
        self.processor.processing_error.connect(self.processing_error)

        self.stop_analysis_btn.setEnabled(True)
        self.mode_label.setText("状态: 录制中 + 分析中")
        self.status_label.setText("正在边录边分析...")
        self.processor.start()

    def start_processing_loaded_video(self):
        if not self.video_path:
            self.status_label.setText("没有可分析的视频")
            return

        if self.is_recording_now():
            self.status_label.setText("当前正在录制，这个按钮只用于分析手动加载的视频")
            return

        if self.processor and self.processor.isRunning():
            return

        self.reset_stats()
        self.processor = self._create_processor(live_mode=False)
        self.processor.frame_processed.connect(self.update_processed_frame)
        self.processor.processing_finished.connect(self.processing_finished)
        self.processor.processing_error.connect(self.processing_error)

        self.start_record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)

        self.mode_label.setText("状态: 分析中")
        self.status_label.setText("正在分析已加载视频...")
        self.processor.start()

    def stop_processing(self):
        if self.processor and self.processor.isRunning():
            self.processor.stop_processing()
            self.status_label.setText("正在停止分析...")

    def update_processed_frame(self, frame, fps, count, processed_frames, available_frames):
        self.current_frame = frame
        self.current_fps = fps
        self.current_count = count
        self.current_processed_frames = processed_frames
        self.current_available_frames = available_frames

        self.display_frame(frame)
        self.fps_label.setText(f"处理FPS: {fps:.1f}")
        self.count_label.setText(f"总计数: {count}")

        if self.is_recording_now():
            self.progress_label.setText(f"处理: {processed_frames} / 已录制: {self.current_recorded_frames}")
            self.mode_label.setText("状态: 录制中 + 分析中")
        else:
            self.progress_label.setText(f"处理: {processed_frames} / 可读总帧: {available_frames}")
            if self.recorder is not None and not self.recorder.isRunning():
                self.mode_label.setText("状态: 录制结束，分析收尾中")
            else:
                self.mode_label.setText("状态: 分析中")

    def processing_finished(self, total_count):
        self.count_label.setText(f"总计数: {total_count}")
        self.progress_label.setText(f"处理完成: {self.current_processed_frames} 帧")

        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(bool(self.video_path) and not self.is_recording_now())
        self.stop_analysis_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

        self.mode_label.setText("状态: 分析完成")

        if self.processor and self.processor.save_output and self.processor.output_path:
            self.status_label.setText(f"分析完成，结果视频已保存到: {self.processor.output_path}")
        else:
            self.status_label.setText("分析完成")

    def processing_error(self, msg):
        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(bool(self.video_path) and not self.is_recording_now())
        self.stop_analysis_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

        self.mode_label.setText("状态: 错误")
        self.status_label.setText(msg)

    def closeEvent(self, event):
        try:
            if self.recorder and self.recorder.isRunning():
                self.recorder.stop()
                self.recorder.wait(1500)
        except Exception:
            pass

        try:
            if self.processor and self.processor.isRunning():
                self.processor.stop_processing()
                self.processor.wait(1500)
        except Exception:
            pass

        event.accept()


# ===================== 入口 =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
