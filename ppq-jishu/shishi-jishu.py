import sys
import os
import cv2
import time
import queue
import threading
import numpy as np
import torch

from collections import defaultdict
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


# ===================== 配置参数 =====================
MODEL_PATH = r'E:\Visual-project\ultralytics-jishu\ppq-jishu\best.pt'
TRACKER_CFG = "bytetrack1.0.yaml"      # 没有这个文件的话改成你本地实际的 tracker 配置
CONF_THRESH = 0.01
TRACK_BUFFER = 180
PROCESS_WIDTH = 640
EMIT_EVERY_N_FRAMES = 1
MAX_TRAIL_SEGMENTS = 100

# -------- 不跳帧模式参数 --------
QUEUE_SIZE = 1000                      # 队列越大，可缓存越多帧，但内存占用也越高
SOURCE_FPS_LIMIT = 15                  # 尝试把摄像头帧率限制到 15，减少积压；RTSP 不一定生效
SAVE_OUTPUT = False                    # 是否保存处理后视频
OUTPUT_PATH = "realtime_no_skip_output.mp4"


class VideoProcessor(QThread):
    frame_processed = pyqtSignal(object, float, int, int, float, int)
    # frame, fps, count, frame_idx, lag_sec, backlog
    processing_finished = pyqtSignal(int)

    def __init__(self, source, count_line):
        super().__init__()
        self.source = source
        self.count_line_orig = count_line

        self.total_count = 0
        self.is_processing = False
        self.processed_frames = 0

        # 不跳帧：队列大 + 阻塞 put
        self.frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.processing_ready = threading.Event()
        self.read_thread = None
        self.capture = None

        self.original_width = 0
        self.original_height = 0
        self.process_width = PROCESS_WIDTH
        self.process_height = 0
        self.video_fps = 30.0

        self.video_writer = None
        self.save_output = SAVE_OUTPUT
        self.output_path = OUTPUT_PATH

        self.performance_stats = {
            'frame_read_time': [],
            'resize_time': [],
            'inference_time': [],
            'tracking_time': [],
            'drawing_time': [],
            'total_frame_time': []
        }

    # -------------------- 输入源工具 --------------------
    def _open_capture(self):
        if isinstance(self.source, int):
            if os.name == "nt":
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source)

        # 这里不再刻意设成极小 buffer，因为你的目标不是“实时低延迟”
        # 但为了尽量稳，还是尝试限制源帧率（不一定每个设备都支持）
        try:
            cap.set(cv2.CAP_PROP_FPS, SOURCE_FPS_LIMIT)
        except Exception:
            pass

        return cap

    def _get_stream_info(self):
        cap = self._open_capture()
        if not cap.isOpened():
            return 0, 0, 0.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0

        if width <= 0 or height <= 0:
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]

        cap.release()

        if fps <= 0 or not np.isfinite(fps):
            fps = float(SOURCE_FPS_LIMIT)

        fps = max(1.0, min(120.0, fps))
        return width, height, fps

    # -------------------- 不跳帧读帧线程 --------------------
    def read_frames_live_no_skip(self):
        try:
            self.processing_ready.wait()

            self.capture = self._open_capture()
            if not self.capture.isOpened():
                print("无法打开实时输入源")
                return

            fail = 0
            frame_index = 0

            while self.is_processing:
                t0 = time.perf_counter()
                ret, frame = self.capture.read()
                read_time = time.perf_counter() - t0

                if not ret or frame is None:
                    fail += 1
                    if fail > 30:
                        break
                    time.sleep(0.01)
                    continue

                fail = 0
                frame_index += 1
                capture_ts = time.time()

                if self.original_width == 0 or self.original_height == 0:
                    self.original_height, self.original_width = frame.shape[:2]

                if self.process_height == 0:
                    self.process_height = int(self.original_height * self.process_width / self.original_width)

                rz_t0 = time.perf_counter()
                process_frame = cv2.resize(frame, (self.process_width, self.process_height))
                resize_time = time.perf_counter() - rz_t0

                item = {
                    "frame": process_frame,
                    "capture_ts": capture_ts,
                    "read_time": read_time,
                    "resize_time": resize_time,
                    "source_frame_idx": frame_index
                }

                # 关键：阻塞 put，不主动丢帧
                while self.is_processing:
                    try:
                        self.frame_queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue

                if frame_index % 200 == 0:
                    print(f"已采集 {frame_index} 帧，当前队列积压: {self.frame_queue.qsize()}")

        except Exception as e:
            print(f"实时读取异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            print("实时读取结束")

    # -------------------- 主线程：推理与绘制 --------------------
    def run(self):
        self.original_width, self.original_height, self.video_fps = self._get_stream_info()
        if self.original_width == 0 or self.original_height == 0:
            print("无法获取实时流信息，提前退出")
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

        print(f"实时流尺寸: {self.original_width}x{self.original_height}")
        print(f"处理尺寸: {self.process_width}x{self.process_height}")
        print(f"处理坐标系计数线: {process_count_line}")
        print(f"输入FPS(估计): {self.video_fps:.2f}")

        print(f"正在加载模型: {MODEL_PATH}")
        t0 = time.time()
        model = YOLO(MODEL_PATH)
        print(f"模型加载耗时: {time.time() - t0:.3f}s")

        warmup = np.zeros((self.process_height, self.process_width, 3), dtype=np.uint8)
        _ = model.predict(
            warmup,
            imgsz=self.process_width,
            conf=CONF_THRESH,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=False
        )

        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, max(1.0, self.video_fps),
                (self.process_width, self.process_height)
            )
            if self.video_writer is None or not self.video_writer.isOpened():
                print("输出视频打开失败，自动关闭保存")
                self.video_writer = None

        self.is_processing = True
        self.read_thread = threading.Thread(
            target=self.read_frames_live_no_skip,
            name="live_reader_no_skip",
            daemon=True
        )
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

        print("开始不跳帧处理")

        try:
            while self.is_processing:
                frame_t0 = time.perf_counter()

                try:
                    item = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.read_thread and not self.read_thread.is_alive():
                        break
                    continue

                frame = item["frame"]
                capture_ts = item["capture_ts"]
                read_time = item["read_time"]
                resize_time = item["resize_time"]

                lag_sec = max(0.0, time.time() - capture_ts)
                backlog = self.frame_queue.qsize()

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
                    imgsz=self.process_width,
                    device=0 if torch.cuda.is_available() else 'cpu'
                )
                inference_time = time.perf_counter() - inf_t0
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
                            track_history[tid]["consecutive_misses"] > 10
                            or track_history[tid]["track_length"] < 5
                        ):
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

                    if (
                        prev is not None
                        and prev > 0
                        and side < 0
                        and hist["track_length"] > 5
                        and not hist["counted"]
                    ):
                        self.total_count += 1
                        hist["counted"] = True

                    if side > 0:
                        hist["counted"] = False

                    hist["prev_side"] = side

                self.performance_stats['tracking_time'].append(time.perf_counter() - trk_t0)

                draw_t0 = time.perf_counter()

                cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)

                for tid, hist in track_history.items():
                    if tid not in current_objs:
                        continue

                    color = hist["color"]
                    x1, y1, x2, y2 = current_objs[tid]['box']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, f'ID:{tid}', (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

                    pts = hist["positions"]
                    start_i = max(1, len(pts) - MAX_TRAIL_SEGMENTS)
                    for i in range(start_i, len(pts)):
                        p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                        p1 = (int(pts[i][0]), int(pts[i][1]))
                        cv2.line(frame, p0, p1, color, 1)

                cv2.putText(frame, f"Total Count: {self.total_count}", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Infer: {inference_time * 1000:.1f}ms", (20, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Active: {active_tracks}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {self.processed_frames}", (20, 134),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Backlog: {backlog}", (20, 158),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.putText(frame, f"Lag: {lag_sec:.2f}s", (20, 182),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                self.performance_stats['drawing_time'].append(time.perf_counter() - draw_t0)
                self.performance_stats['total_frame_time'].append(time.perf_counter() - frame_t0)

                if self.video_writer is not None:
                    self.video_writer.write(frame)

                if (self.processed_frames % EMIT_EVERY_N_FRAMES) == 0:
                    self.frame_processed.emit(
                        frame, current_fps, self.total_count,
                        self.processed_frames, lag_sec, backlog
                    )

                if self.processed_frames % 200 == 0:
                    self.print_performance_stats()

        except Exception as e:
            print(f"处理异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False

            if self.read_thread and self.read_thread.is_alive():
                try:
                    self.read_thread.join(timeout=2.0)
                except Exception:
                    pass

            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None

            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass

            print("\n=== 最终性能统计 ===")
            self.print_performance_stats()
            print(f"处理结束! 处理帧数: {self.processed_frames}, 总计数: {self.total_count}")
            self.processing_finished.emit(self.total_count)

    def print_performance_stats(self):
        if self.processed_frames == 0:
            return

        print(f"\n=== 第 {self.processed_frames} 帧性能统计 ===")
        stats = [
            ('帧读取', 'frame_read_time', 1000),
            ('图像缩放', 'resize_time', 1000),
            ('模型推理', 'inference_time', 1000),
            ('目标跟踪', 'tracking_time', 1000),
            ('绘制显示', 'drawing_time', 1000),
            ('单帧总耗时', 'total_frame_time', 1000)
        ]

        for name, key, mult in stats:
            arr = self.performance_stats.get(key, [])
            if arr:
                recent = arr[-50:]
                if recent:
                    avg = np.mean(recent) * mult
                    mn = np.min(recent) * mult
                    mx = np.max(recent) * mult
                    print(f"{name}: 平均{avg:.2f}ms, 最小{mn:.2f}ms, 最大{mx:.2f}ms")

        total_arr = self.performance_stats['total_frame_time']
        total_avg = np.mean(total_arr[-50:]) if total_arr else 0
        if total_avg > 0:
            print("\n各环节时间占比:")
            for name, key, _ in stats[:-1]:
                arr = self.performance_stats.get(key, [])
                if arr:
                    recent = arr[-50:]
                    if recent:
                        pct = (np.mean(recent) / total_avg) * 100
                        print(f"  {name}: {pct:.1f}%")

    def stop_processing(self):
        self.is_processing = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = None
        self.source = 0
        self.count_line = [300, 2, 300, 600]
        self.current_frame = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("乒乓球实时计数系统（不跳帧延迟模式）")
        self.setGeometry(100, 100, 1250, 850)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        source_layout = QHBoxLayout()
        self.source_edit = QLineEdit("0")
        self.source_edit.setPlaceholderText("输入摄像头编号(0/1) 或 RTSP 地址")
        source_layout.addWidget(self.source_edit)

        self.preview_btn = QPushButton("打开预览")
        self.preview_btn.clicked.connect(self.preview_source)
        source_layout.addWidget(self.preview_btn)
        main_layout.addLayout(source_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 550)
        self.video_label.setText("实时预览区域")
        self.video_label.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.video_label)

        info_layout = QHBoxLayout()

        self.fps_label = QLabel("处理FPS: 0.0")
        self.fps_label.setStyleSheet("font-weight: bold; color: green;")
        info_layout.addWidget(self.fps_label)

        self.count_label = QLabel("总计数: 0")
        self.count_label.setStyleSheet("font-weight: bold; color: blue;")
        info_layout.addWidget(self.count_label)

        self.frame_label = QLabel("已处理帧: 0")
        self.frame_label.setStyleSheet("font-weight: bold; color: orange;")
        info_layout.addWidget(self.frame_label)

        self.backlog_label = QLabel("队列积压: 0")
        self.backlog_label.setStyleSheet("font-weight: bold; color: purple;")
        info_layout.addWidget(self.backlog_label)

        self.lag_label = QLabel("输出延迟: 0.00s")
        self.lag_label.setStyleSheet("font-weight: bold; color: brown;")
        info_layout.addWidget(self.lag_label)

        main_layout.addLayout(info_layout)

        btn_layout = QHBoxLayout()
        self.process_btn = QPushButton("开始检测")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        btn_layout.addWidget(self.process_btn)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_processing)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        self.status_label = QLabel("请输入摄像头编号或 RTSP 地址，然后打开预览")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.status_label)

    def get_input_source(self):
        text = self.source_edit.text().strip()
        if text == "":
            return 0
        if text.isdigit():
            return int(text)
        return text

    def preview_source(self):
        source = self.get_input_source()

        if isinstance(source, int):
            if os.name == "nt":
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        try:
            cap.set(cv2.CAP_PROP_FPS, SOURCE_FPS_LIMIT)
        except Exception:
            pass

        if not cap.isOpened():
            self.status_label.setText("无法打开输入源")
            cap.release()
            return

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self.status_label.setText("无法读取输入源画面")
            return

        self.source = source
        self.current_frame = frame
        self.display_frame(frame)

        h, w = frame.shape[:2]
        self.count_line = [w // 2, int(h * 0.01), w // 2, int(h * 0.99)]

        self.process_btn.setEnabled(True)
        self.status_label.setText(f"预览成功 - 输入尺寸: {w}x{h}")

    def start_processing(self):
        source = self.get_input_source()
        self.source = source

        self.status_label.setText("检测中（不跳帧，允许延迟）...")
        self.process_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.source_edit.setEnabled(False)

        self.count_label.setText("总计数: 0")
        self.frame_label.setText("已处理帧: 0")
        self.backlog_label.setText("队列积压: 0")
        self.lag_label.setText("输出延迟: 0.00s")

        self.processor = VideoProcessor(self.source, self.count_line)
        self.processor.frame_processed.connect(self.update_processed_frame)
        self.processor.processing_finished.connect(self.processing_finished)
        self.processor.start()

    def processing_finished(self, total_count):
        self.status_label.setText("检测已停止")
        self.process_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.source_edit.setEnabled(True)
        self.count_label.setText(f"总计数: {total_count}")

    def stop_processing(self):
        if self.processor and self.processor.isRunning():
            self.processor.stop_processing()
            self.processor.wait(2000)

        self.status_label.setText("检测已停止")
        self.process_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.source_edit.setEnabled(True)

    def update_processed_frame(self, frame, fps, count, frame_idx, lag_sec, backlog):
        self.current_frame = frame
        self.display_frame(frame)

        self.fps_label.setText(f"处理FPS: {fps:.1f}")
        self.count_label.setText(f"总计数: {count}")
        self.frame_label.setText(f"已处理帧: {frame_idx}")
        self.backlog_label.setText(f"队列积压: {backlog}")
        self.lag_label.setText(f"输出延迟: {lag_sec:.2f}s")

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())