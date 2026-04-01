import sys
import os
import cv2
import time
import numpy as np
import threading
import queue
import subprocess
from collections import defaultdict

import torch
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap


# ===================== 全局配置 =====================
# 这里改成你自己的模型路径
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


# ===================== 录制线程 =====================
class RecorderThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, float, int)              # frame, fps, recorded_frames
    recording_started = pyqtSignal(str, int, int, float)         # path, width, height, fps
    recording_finished = pyqtSignal(str, int)                    # path, total_frames
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
        self.output_file = f"output_{timestamp}.mp4"

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        if not cap.isOpened():
            self.recording_error.emit("无法打开摄像头")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

        if actual_width <= 0 or actual_height <= 0:
            cap.release()
            self.recording_error.emit("无法获取摄像头分辨率")
            return

        if actual_fps <= 0 or not np.isfinite(actual_fps):
            actual_fps = self.target_fps

        # mp4v 兼容性通常更稳
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            cap.release()
            out.release()
            self.recording_finished.emit(self.output_file, recorded_frames)


# ===================== 分析线程（沿用你的 jishu 主体逻辑） =====================
class VideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray, float, int, int)    # frame, fps, count, progress
    processing_finished = pyqtSignal(int)
    processing_error = pyqtSignal(str)

    def __init__(self, video_path, count_line):
        super().__init__()
        self.video_path = video_path
        self.count_line_orig = count_line
        self.total_count = 0
        self.is_processing = False

        self.frame_queue = queue.Queue(maxsize=500)
        self.processing_ready = threading.Event()
        self.read_thread = None

        self.original_width = 0
        self.original_height = 0
        self.total_frames = 0

        self.process_width = PROCESS_WIDTH
        self.process_height = 0

        self.performance_stats = {
            'frame_read_time': [],
            'resize_time': [],
            'inference_time': [],
            'tracking_time': [],
            'drawing_time': [],
            'total_frame_time': []
        }

        self.ffmpeg_process = None

        self.save_output = True
        self.output_path = ""
        self.video_writer = None
        self.video_fps = 25.0

    def _check_ffmpeg_available(self):
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False

    def _check_ffprobe_available(self):
        try:
            result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False

    def _get_video_info(self):
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_frames',
                '-of', 'csv=p=0',
                self.video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    width = int(parts[0])
                    height = int(parts[1])
                    frames = int(parts[2]) if parts[2].isdigit() else 0
                    return width, height, frames
        except Exception:
            pass

        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return width, height, frames

        return 0, 0, 0

    def _get_video_fps(self):
        fps = 0.0

        if self._check_ffprobe_available():
            try:
                cmd = [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=avg_frame_rate',
                    '-of', 'default=nw=1:nk=1',
                    self.video_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    rate = result.stdout.strip()
                    if '/' in rate:
                        num, den = rate.split('/')
                        num = float(num)
                        den = float(den) if float(den) != 0 else 1.0
                        fps = num / den
                    elif rate:
                        fps = float(rate)
            except Exception:
                fps = 0.0

        if fps <= 0 or not np.isfinite(fps):
            try:
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                cap.release()
            except Exception:
                fps = 0.0

        if fps <= 0 or not np.isfinite(fps):
            fps = 25.0

        fps = max(1.0, min(120.0, fps))
        return fps

    def read_frames_ffmpeg(self):
        try:
            self.processing_ready.wait()

            if self.original_width == 0 or self.original_height == 0:
                self.original_width, self.original_height, self.total_frames = self._get_video_info()
                if self.original_width == 0 or self.original_height == 0:
                    return

            self.process_height = int(self.original_height * self.process_width / self.original_width)
            vf = f"scale={self.process_width}:{self.process_height}"

            ffmpeg_cmd = [
                "ffmpeg", "-i", self.video_path,
                "-an", "-sn",
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-threads", "0",
                "-vf", vf,
                "-pix_fmt", "bgr24",
                "-f", "rawvideo",
                "-vsync", "0",
                "-loglevel", "quiet",
                "-"
            ]

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 ** 8
            )

            frame_size = self.process_width * self.process_height * 3
            consecutive_failures = 0

            while self.is_processing:
                t0 = time.perf_counter()
                raw = self.ffmpeg_process.stdout.read(frame_size)
                if not raw or len(raw) == 0:
                    break

                if len(raw) != frame_size:
                    consecutive_failures += 1
                    if consecutive_failures > 5:
                        break
                    continue

                consecutive_failures = 0
                read_time = time.perf_counter() - t0

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.process_height, self.process_width, 3)
                )
                self.frame_queue.put((frame, read_time, 0.0))

        except Exception:
            pass
        finally:
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                except Exception:
                    pass
                try:
                    if self.ffmpeg_process.stdout:
                        self.ffmpeg_process.stdout.close()
                    if self.ffmpeg_process.stderr:
                        self.ffmpeg_process.stderr.close()
                except Exception:
                    pass
                try:
                    self.ffmpeg_process.wait(timeout=2)
                except Exception:
                    try:
                        self.ffmpeg_process.kill()
                    except Exception:
                        pass
                self.ffmpeg_process = None

    def read_frames_fallback(self):
        self.processing_ready.wait()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.original_width == 0 or self.original_height == 0:
            self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.process_height = int(self.original_height * self.process_width / self.original_width)

        fail = 0
        while self.is_processing and fail < 10:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                fail += 1
                time.sleep(0.001)
                continue

            fail = 0

            rz_t0 = time.perf_counter()
            process_frame = cv2.resize(frame, (self.process_width, self.process_height))
            rz_t1 = time.perf_counter()

            read_time = rz_t0 - t0
            resize_time = rz_t1 - rz_t0

            self.frame_queue.put((process_frame, read_time, resize_time))

        cap.release()

    def run(self):
        try:
            self.original_width, self.original_height, self.total_frames = self._get_video_info()
            if self.original_width == 0 or self.original_height == 0:
                self.processing_error.emit("无法获取视频信息")
                self.processing_finished.emit(0)
                return

            self.process_height = int(self.original_height * self.process_width / self.original_width)

            sx = self.process_width / self.original_width
            sy = self.process_height / self.original_height
            process_count_line = [
                int(self.count_line_orig[0] * sx),
                int(self.count_line_orig[1] * sy),
                int(self.count_line_orig[2] * sx),
                int(self.count_line_orig[3] * sy)
            ]

            self.video_fps = self._get_video_fps()

            model = YOLO(MODEL_PATH)

            warmup = np.zeros((self.process_height, self.process_width, 3), dtype=np.uint8)
            _ = model.predict(
                warmup,
                imgsz=self.process_width,
                conf=CONF_THRESH,
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=False
            )

            if self.save_output:
                base, _ext = os.path.splitext(self.video_path)
                self.output_path = base + "_analyzed.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_path, fourcc, self.video_fps,
                    (self.process_width, self.process_height)
                )
                if self.video_writer is not None and not self.video_writer.isOpened():
                    self.video_writer = None

            self.is_processing = True

            if self._check_ffmpeg_available():
                self.read_thread = threading.Thread(target=self.read_frames_ffmpeg, daemon=True)
            else:
                self.read_thread = threading.Thread(target=self.read_frames_fallback, daemon=True)

            self.read_thread.start()
            self.processing_ready.set()

            self.total_count = 0
            self.processed_frames = 0
            fps_counter = 0
            fps_t0 = time.time()
            current_fps = 0.0

            track_history = defaultdict(lambda: {
                "positions": [],
                "prev_side": None,
                "counted": False,
                "last_seen": 0,
                "consecutive_misses": 0,
                "track_length": 0,
                "color": tuple(np.random.randint(0, 255, 3).tolist())
            })

            def smooth_positions(positions, window_size=3):
                if len(positions) < window_size:
                    return positions[-1] if positions else None
                return np.mean(positions[-window_size:], axis=0)

            while self.is_processing:
                frame_t0 = time.perf_counter()

                try:
                    frame, read_time, resize_time = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.read_thread and not self.read_thread.is_alive():
                        break
                    continue

                self.performance_stats['frame_read_time'].append(read_time)
                self.performance_stats['resize_time'].append(resize_time)

                self.processed_frames += 1
                fps_counter += 1
                now = time.time()
                if now - fps_t0 >= 1.0:
                    current_fps = fps_counter / (now - fps_t0)
                    fps_counter = 0
                    fps_t0 = now

                inf_t0 = time.perf_counter()
                results = model.track(
                    frame,
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
                        if track_history[tid]["consecutive_misses"] > 10 or track_history[tid]["track_length"] < 5:
                            del track_history[tid]
                        else:
                            active_tracks += 1

                (lx1, ly1, lx2, ly2) = process_count_line
                A = ly2 - ly1
                B = lx1 - lx2
                C = lx2 * ly1 - lx1 * ly2

                for tid, hist in list(track_history.items()):
                    if tid not in current_objs:
                        continue

                    pos = hist["positions"]
                    if len(pos) < 2:
                        continue

                    sp = smooth_positions(pos)
                    if sp is None:
                        continue

                    side = np.sign(A * sp[0] + B * sp[1] + C)
                    prev = hist["prev_side"]

                    if (prev is not None and prev > 0 and side < 0 and
                            hist["track_length"] > 5 and not hist["counted"]):
                        self.total_count += 1
                        hist["counted"] = True

                    if side > 0:
                        hist["counted"] = False

                    hist["prev_side"] = side

                trk_t1 = time.perf_counter()
                self.performance_stats['tracking_time'].append(trk_t1 - trk_t0)

                draw_t0 = time.perf_counter()
                cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)

                for tid, hist in track_history.items():
                    if tid not in current_objs:
                        continue

                    color = hist["color"]
                    x1, y1, x2, y2 = current_objs[tid]['box']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID:{tid}', (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    pts = hist["positions"]
                    start_i = max(1, len(pts) - MAX_TRAIL_SEGMENTS)
                    for i in range(start_i, len(pts)):
                        p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                        p1 = (int(pts[i][0]), int(pts[i][1]))
                        cv2.line(frame, p0, p1, color, 1)

                progress = (self.processed_frames / self.total_frames) * 100 if self.total_frames > 0 else 0

                cv2.putText(frame, f"Total Count: {self.total_count}", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Infer: {inference_time * 1000:.1f}ms", (20, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Active: {active_tracks}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Progress: {progress:.1f}%", (20, 134),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                draw_t1 = time.perf_counter()
                self.performance_stats['drawing_time'].append(draw_t1 - draw_t0)
                self.performance_stats['total_frame_time'].append(time.perf_counter() - frame_t0)

                if self.video_writer is not None:
                    self.video_writer.write(frame)

                if (self.processed_frames % EMIT_EVERY_N_FRAMES) == 0:
                    self.frame_processed.emit(frame, current_fps, self.total_count, int(progress))

        except Exception as e:
            self.processing_error.emit(f"分析异常: {e}")
        finally:
            self.is_processing = False

            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                except Exception:
                    pass
                try:
                    if self.ffmpeg_process.stdout:
                        self.ffmpeg_process.stdout.close()
                    if self.ffmpeg_process.stderr:
                        self.ffmpeg_process.stderr.close()
                except Exception:
                    pass
                try:
                    self.ffmpeg_process.wait(timeout=1)
                except Exception:
                    try:
                        self.ffmpeg_process.kill()
                    except Exception:
                        pass
                self.ffmpeg_process = None

            if self.read_thread and self.read_thread.is_alive():
                try:
                    self.read_thread.join(timeout=2.0)
                except Exception:
                    pass

            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass

            self.processing_finished.emit(self.total_count)

    def stop_processing(self):
        self.is_processing = False
        try:
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
        except Exception:
            pass

        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                if self.ffmpeg_process.stdout:
                    self.ffmpeg_process.stdout.close()
                if self.ffmpeg_process.stderr:
                    self.ffmpeg_process.stderr.close()
            except Exception:
                pass


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
        self.current_progress = 0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("乒乓球录制 + 自动计数分析一体化界面")
        self.setGeometry(100, 100, 1280, 860)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("录制后的视频路径会显示在这里，也可以手动加载已有视频")
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

        self.progress_label = QLabel("进度: 0%")
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
        self.start_analysis_btn.clicked.connect(self.start_processing)
        self.start_analysis_btn.setEnabled(False)
        btn_layout.addWidget(self.start_analysis_btn)

        self.stop_analysis_btn = QPushButton("停止分析")
        self.stop_analysis_btn.clicked.connect(self.stop_processing)
        self.stop_analysis_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_analysis_btn)

        main_layout.addLayout(btn_layout)

        self.status_label = QLabel("点击“开始录制”，录制结束后会自动开始分析")
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
                self.count_line = [w // 2, int(h * 0.01), w // 2, int(h * 0.99)]
            cap.release()

    # ---------- 手动加载视频 ----------
    def browse_video_file(self):
        if self.recorder and self.recorder.isRunning():
            self.status_label.setText("录制中，不能加载其它视频")
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
            self.status_label.setText("视频已加载，可以直接开始分析")
            self.mode_label.setText("状态: 已加载视频")

    # ---------- 录制 ----------
    def start_recording(self):
        if self.processor and self.processor.isRunning():
            self.status_label.setText("分析进行中，不能开始录制")
            return

        self.video_path = ""
        self.file_path_edit.clear()
        self.current_count = 0
        self.current_progress = 0
        self.count_label.setText("总计数: 0")
        self.progress_label.setText("进度: 0%")

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
        self.status_label.setText("正在录制...")
        self.recorder.start()

    def stop_recording(self):
        if self.recorder and self.recorder.isRunning():
            self.status_label.setText("正在停止录制...")
            self.recorder.stop()

    def on_recording_started(self, path, width, height, fps):
        self.video_path = path
        self.file_path_edit.setText(path)
        self.count_line = [width // 2, int(height * 0.01), width // 2, int(height * 0.99)]
        self.status_label.setText(f"录制开始：{width}x{height} @ {fps:.2f}fps")

    def on_recording_finished(self, path, total_frames):
        self.video_path = path
        self.file_path_edit.setText(path)

        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

        if total_frames <= 0:
            self.mode_label.setText("状态: 待机")
            self.status_label.setText("录制失败或没有录到有效帧")
            self.start_analysis_btn.setEnabled(False)
            return

        self.preview_video_first_frame(path)
        self.start_analysis_btn.setEnabled(True)

        self.mode_label.setText("状态: 录制完成")
        self.status_label.setText(f"录制完成，共 {total_frames} 帧，马上开始分析...")

        # 自动开始分析
        QTimer.singleShot(500, self.start_processing)

    def on_recording_error(self, msg):
        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.start_analysis_btn.setEnabled(False)

        self.mode_label.setText("状态: 错误")
        self.status_label.setText(msg)

    def update_recording_frame(self, frame, fps, recorded_frames):
        self.current_frame = frame
        self.display_frame(frame)
        self.fps_label.setText(f"录制FPS: {fps:.1f}")
        self.count_label.setText("总计数: 0")
        self.progress_label.setText(f"已录制: {recorded_frames} 帧")

    # ---------- 分析 ----------
    def start_processing(self):
        if not self.video_path:
            self.status_label.setText("没有可分析的视频")
            return

        if self.recorder and self.recorder.isRunning():
            self.status_label.setText("请先停止录制")
            return

        if self.processor and self.processor.isRunning():
            return

        self.processor = VideoProcessor(self.video_path, self.count_line)
        self.processor.frame_processed.connect(self.update_processed_frame)
        self.processor.processing_finished.connect(self.processing_finished)
        self.processor.processing_error.connect(self.processing_error)

        self.start_record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)

        self.current_count = 0
        self.count_label.setText("总计数: 0")
        self.mode_label.setText("状态: 分析中")
        self.status_label.setText("正在分析视频...")
        self.processor.start()

    def stop_processing(self):
        if self.processor and self.processor.isRunning():
            self.processor.stop_processing()
            self.status_label.setText("正在停止分析...")

    def update_processed_frame(self, frame, fps, count, progress):
        self.current_frame = frame
        self.current_fps = fps
        self.current_count = count
        self.current_progress = progress

        self.display_frame(frame)
        self.fps_label.setText(f"处理FPS: {fps:.1f}")
        self.count_label.setText(f"总计数: {count}")
        self.progress_label.setText(f"进度: {progress}%")

    def processing_finished(self, total_count):
        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

        self.count_label.setText(f"总计数: {total_count}")
        self.progress_label.setText("进度: 100%")
        self.mode_label.setText("状态: 分析完成")

        if self.processor and self.processor.save_output and self.processor.output_path:
            self.status_label.setText(f"分析完成，结果视频已保存到: {self.processor.output_path}")
        else:
            self.status_label.setText("分析完成")

    def processing_error(self, msg):
        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

        self.mode_label.setText("状态: 错误")
        self.status_label.setText(msg)

    def closeEvent(self, event):
        try:
            if self.recorder and self.recorder.isRunning():
                self.recorder.stop()
                self.recorder.wait(1000)
        except Exception:
            pass

        try:
            if self.processor and self.processor.isRunning():
                self.processor.stop_processing()
                self.processor.wait(1000)
        except Exception:
            pass

        event.accept()


# ===================== 入口 =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())