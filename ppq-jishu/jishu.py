import sys
import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import time
import threading
import queue
import subprocess

# ===================== 配置参数（匹配 TRT 320×320） =====================
MODEL_PATH = r'E:\Visual-project\ultralytics-jishu\ppq-jishu\best.pt'
CONF_THRESH = 0.01
TRACK_BUFFER = 180
PROCESS_WIDTH = 500            # 宽 320；Ultralytics 会 letterbox 到 320×320
EMIT_EVERY_N_FRAMES = 1
MAX_TRAIL_SEGMENTS = 100

class VideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray, float, int, int)
    processing_finished = pyqtSignal(int)

    def __init__(self, video_path, count_line):
        super().__init__()
        self.video_path = video_path
        self.count_line_orig = count_line
        self.total_count = 0
        self.is_processing = False

        # ★ 队列调大 + 热身就绪信号
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

        # ====== 新增：视频输出相关 ======
        self.save_output = True                  # 是否保存输出视频
        self.output_path = ""                    # 输出路径（运行时自动生成）
        self.video_writer = None                 # cv2.VideoWriter
        self.video_fps = 25.0                    # 源视频帧率（运行时自动探测）

    # -------------------- 工具 --------------------
    def _check_ffmpeg_available(self):
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False

    def _check_ffprobe_available(self):
        try:
            result = subprocess.run(['ffprobe', '-version'],
                                    capture_output=True, text=True, timeout=2)
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
                width, height, frames = result.stdout.strip().split(',')
                width = int(width); height = int(height)
                frames = int(frames) if frames and frames.isdigit() else 0
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

    # ====== 新增：获取视频 FPS ======
    def _get_video_fps(self):
        fps = 0.0
        # 优先使用 ffprobe
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
                    rate = result.stdout.strip()  # 如 "30000/1001" 或 "25/1"
                    if '/' in rate:
                        num, den = rate.split('/')
                        num = float(num); den = float(den) if float(den) != 0 else 1.0
                        fps = num / den
                    else:
                        fps = float(rate)
            except Exception:
                fps = 0.0

        # 回退到 OpenCV
        if fps <= 0 or fps != fps:
            try:
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                cap.release()
            except Exception:
                fps = 0.0

        # 兜底
        if fps <= 0 or not np.isfinite(fps):
            fps = 25.0

        # 限幅，避免异常值
        fps = max(1.0, min(120.0, fps))
        print(f"检测到源视频 FPS: {fps:.3f}")
        return fps

    # -------------------- FFmpeg：整帧缩放到宽 320 → rawvideo --------------------
    def read_frames_ffmpeg(self):
        try:
            # ★ 等待处理端热身完毕
            self.processing_ready.wait()

            if self.original_width == 0 or self.original_height == 0:
                self.original_width, self.original_height, self.total_frames = self._get_video_info()
                if self.original_width == 0 or self.original_height == 0:
                    print("FFmpeg读取: 无法获取视频信息")
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
            print("FFmpeg 命令：", " ".join(ffmpeg_cmd))
            print(f"原始尺寸: {self.original_width}x{self.original_height} | 处理尺寸: {self.process_width}x{self.process_height}")

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
            )

            frame_size = self.process_width * self.process_height * 3
            read_frames = 0
            consecutive_failures = 0

            while self.is_processing:
                t0 = time.perf_counter()
                raw = self.ffmpeg_process.stdout.read(frame_size)
                if not raw or len(raw) == 0:
                    break
                if len(raw) != frame_size:
                    consecutive_failures += 1
                    if consecutive_failures > 5: break
                    continue
                consecutive_failures = 0
                read_time = time.perf_counter() - t0

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.process_height, self.process_width, 3)
                )

                # ★ 阻塞 put，不丢帧
                self.frame_queue.put((frame, read_time, 0.0))

                read_frames += 1
                if read_frames % 200 == 0:
                    print(f"FFmpeg 已读取 {read_frames} 帧")

        except Exception as e:
            print(f"FFmpeg 读取异常: {e}")
        finally:
            if self.ffmpeg_process:
                try: self.ffmpeg_process.terminate()
                except Exception: pass
                try:
                    if self.ffmpeg_process.stdout: self.ffmpeg_process.stdout.close()
                    if self.ffmpeg_process.stderr: self.ffmpeg_process.stderr.close()
                except Exception: pass
                try: self.ffmpeg_process.wait(timeout=2)
                except Exception:
                    try: self.ffmpeg_process.kill()
                    except Exception: pass
                self.ffmpeg_process = None
            print("FFmpeg 读取结束")

    # -------------------- OpenCV 回退：整帧缩放到宽 320 --------------------
    def read_frames_fallback(self):
        print("未检测到 FFmpeg，自动回退到 OpenCV 读取模式（性能稍弱）")

        # ★ 等待处理端热身完毕
        self.processing_ready.wait()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("OpenCV 无法打开视频文件"); return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.original_width == 0 or self.original_height == 0:
            self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.process_height = int(self.original_height * self.process_width / self.original_width)
        print(f"OpenCV 回退 | 原始尺寸: {self.original_width}x{self.original_height} | 处理尺寸: {self.process_width}x{self.process_height}")

        read_frames = 0
        fail = 0
        while self.is_processing and fail < 10:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                fail += 1; time.sleep(0.001); continue
            fail = 0

            rz_t0 = time.perf_counter()
            process_frame = cv2.resize(frame, (self.process_width, self.process_height))
            rz_t1 = time.perf_counter()

            read_time = rz_t0 - t0
            resize_time = rz_t1 - rz_t0

            # ★ 阻塞 put，不丢帧
            self.frame_queue.put((process_frame, read_time, resize_time))

            read_frames += 1
            if read_frames % 200 == 0:
                print(f"OpenCV 已读取 {read_frames} 帧")

        cap.release()
        print("OpenCV 回退读取结束")

    # -------------------- 主线程：推理与绘制（整帧） --------------------
    def run(self):
        # 1) 获取视频信息
        self.original_width, self.original_height, self.total_frames = self._get_video_info()
        if self.original_width == 0 or self.original_height == 0:
            print("无法获取视频信息，提前退出")
            self.processing_finished.emit(0); return

        # 2) 计算处理尺寸 & 映射计数线（到处理坐标系：宽=320，高按比例）
        self.process_height = int(self.original_height * self.process_width / self.original_width)
        sx = self.process_width / self.original_width
        sy = self.process_height / self.original_height
        process_count_line = [
            int(self.count_line_orig[0] * sx),
            int(self.count_line_orig[1] * sy),
            int(self.count_line_orig[2] * sx),
            int(self.count_line_orig[3] * sy)
        ]
        print(f"处理尺寸: {self.process_width}x{self.process_height} | 处理坐标系计数线: {process_count_line}")

        # 2.1) 获取源视频 FPS（用于输出）
        self.video_fps = self._get_video_fps()

        # 3) 加载模型
        print(f"正在加载 YOLO TensorRT 引擎: {MODEL_PATH}")
        t0 = time.time()
        model = YOLO(MODEL_PATH)
        print(f"模型加载耗时: {time.time() - t0:.3f}s")

        # 4) ★★★ 热身（在读取线程启动前完成）
        warmup = np.zeros((self.process_height, self.process_width, 3), dtype=np.uint8)
        _ = model.predict(
            warmup,
            imgsz=self.process_width,          # 320
            conf=CONF_THRESH,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=False
        )

        # ====== 新增：初始化视频写入器 ======
        if self.save_output:
            base, _ext = os.path.splitext(self.video_path)
            self.output_path = base + "_analyzed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.video_fps, (self.process_width, self.process_height)
            )
            if self.video_writer is None or not self.video_writer.isOpened():
                print("警告：VideoWriter 打开失败，将不保存输出视频。")
                self.video_writer = None
            else:
                print(f"输出视频将保存到: {self.output_path} | FPS: {self.video_fps:.3f} | 尺寸: {self.process_width}x{self.process_height}")

        # 5) 启动读取线程（FFmpeg 优先），但读取线程已在 wait()，这里放行
        self.is_processing = True
        if self._check_ffmpeg_available():
            self.read_thread = threading.Thread(target=self.read_frames_ffmpeg, name="ffmpeg_reader", daemon=True)
            print("使用 FFmpeg 直接读取模式（整帧缩放）")
        else:
            self.read_thread = threading.Thread(target=self.read_frames_fallback, name="cv_reader", daemon=True)
            print("未检测到 FFmpeg，自动回退到 OpenCV 读取模式（性能稍弱）")
        self.read_thread.start()

        # ★ 放行读取线程
        self.processing_ready.set()

        # 6) 主循环
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

        print("开始处理（整帧）")

        try:
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
                    fps_counter = 0; fps_t0 = now

                inf_t0 = time.perf_counter()
                results = model.track(
                    frame,
                    persist=True,
                    conf=CONF_THRESH,
                    tracker="bytetrack1.0.yaml",
                    verbose=False,
                    imgsz=640,                       # 与 TRT 引擎一致
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
                        current_objs[tid] = {'center': (cx, cy), 'box': box, 'confidence': conf}

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
                        if (track_history[tid]["consecutive_misses"] > 10 or
                                track_history[tid]["track_length"] < 5):
                            del track_history[tid]
                        else:
                            active_tracks += 1

                (lx1, ly1, lx2, ly2) = process_count_line
                A = ly2 - ly1; B = lx1 - lx2; C = lx2 * ly1 - lx1 * ly2

                for tid, hist in list(track_history.items()):
                    if tid not in current_objs: continue
                    pos = hist["positions"]
                    if len(pos) < 2: continue
                    sp = smooth_positions(pos)
                    if sp is None: continue
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
                    if tid not in current_objs: continue
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
                cv2.putText(frame, f"Infer: {inference_time*1000:.1f}ms", (20, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Active: {active_tracks}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Size: {self.process_width}x{self.process_height}", (20, 134),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Progress: {progress:.1f}%", (20, 158),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                draw_t1 = time.perf_counter()
                self.performance_stats['drawing_time'].append(draw_t1 - draw_t0)
                self.performance_stats['total_frame_time'].append(time.perf_counter() - frame_t0)

                # ====== 新增：写入输出视频帧 ======
                if self.video_writer is not None:
                    self.video_writer.write(frame)

                if (self.processed_frames % EMIT_EVERY_N_FRAMES) == 0:
                    self.frame_processed.emit(frame, current_fps, self.total_count, int(progress))

                if self.processed_frames % 200 == 0:
                    self.print_performance_stats()

        except Exception as e:
            print(f"处理异常: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.is_processing = False
            if self.ffmpeg_process:
                try: self.ffmpeg_process.terminate()
                except Exception: pass
                try:
                    if self.ffmpeg_process.stdout: self.ffmpeg_process.stdout.close()
                    if self.ffmpeg_process.stderr: self.ffmpeg_process.stderr.close()
                except Exception: pass
                try: self.ffmpeg_process.wait(timeout=1)
                except Exception:
                    try: self.ffmpeg_process.kill()
                    except Exception: pass
                self.ffmpeg_process = None
            if self.read_thread and self.read_thread.is_alive():
                try: self.read_thread.join(timeout=2.0)
                except Exception: pass

            # ====== 新增：释放 VideoWriter ======
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                print(f"结果视频已保存到: {self.output_path}")

            print("\n=== 最终性能统计 ===")
            self.print_performance_stats()
            print(f"处理完成! 处理帧数: {self.processed_frames}, 总计数: {self.total_count}")
            self.processing_finished.emit(self.total_count)

    def print_performance_stats(self):
        if self.processed_frames == 0: return
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
        try:
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
        except Exception:
            pass
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                if self.ffmpeg_process.stdout: self.ffmpeg_process.stdout.close()
                if self.ffmpeg_process.stderr: self.ffmpeg_process.stderr.close()
            except Exception:
                pass

# ===================== PyQt5 UI =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.processor = None
        self.video_path = ""
        self.count_line = [300, 2, 300, 600]
        self.current_frame = None
        self.current_fps = 0.0
        self.current_count = 0
        self.current_progress = 0

    def initUI(self):
        self.setWindowTitle("视频检测系统 - 整帧检测（TRT 热身后再读）")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("请选择视频文件...")
        file_layout.addWidget(self.file_path_edit)
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_video_file)
        file_layout.addWidget(self.browse_btn)
        main_layout.addLayout(file_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("视频预览区域")
        self.video_label.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.video_label)

        info_layout = QHBoxLayout()
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
        self.process_btn = QPushButton("开始检测")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        btn_layout.addWidget(self.process_btn)
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_processing)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        self.status_label = QLabel("请选择视频文件")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.status_label)

    def browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)"
        )
        if file_path:
            self.video_path = file_path
            self.file_path_edit.setText(file_path)
            self.process_btn.setEnabled(True)
            self.status_label.setText("视频文件已选择，可直接开始检测")
            self.preview_video_first_frame(file_path)

    def preview_video_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                h, w = frame.shape[:2]
                self.count_line = [w // 2, int(h * 0.01), w // 2, int(h * 0.99)]
                self.status_label.setText(f"视频预览就绪 - 尺寸: {w}x{h}")
            cap.release()

    def start_processing(self):
        if not self.video_path:
            self.status_label.setText("错误：请先选择视频文件"); return
        self.status_label.setText("检测中...")
        self.process_btn.setEnabled(False); self.browse_btn.setEnabled(False)
        self.current_count = 0; self.count_label.setText("总计数: 0")

        self.processor = VideoProcessor(self.video_path, self.count_line)
        self.processor.frame_processed.connect(self.update_processed_frame)
        self.processor.processing_finished.connect(self.processing_finished)
        self.processor.start()

    def processing_finished(self, total_count):
        # 显示完成状态 + 输出路径（若保存开启）
        if self.processor and self.processor.save_output and self.processor.output_path:
            self.status_label.setText(f"检测完成，结果视频已保存到: {self.processor.output_path}")
        else:
            self.status_label.setText("检测完成")
        self.process_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.count_label.setText(f"总计数: {total_count}")
        self.progress_label.setText(f"进度: 100%")

    def stop_processing(self):
        if self.processor and self.processor.isRunning():
            self.processor.stop_processing()
            self.status_label.setText("检测已停止")
            self.process_btn.setEnabled(True)
            self.browse_btn.setEnabled(True)

    def update_processed_frame(self, frame, fps, count, progress):
        self.current_frame = frame
        self.current_fps = fps
        self.current_count = count
        self.current_progress = progress
        self.display_frame(frame)
        self.fps_label.setText(f"处理FPS: {fps:.1f}")
        self.count_label.setText(f"总计数: {count}")
        self.progress_label.setText(f"进度: {progress}%")

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
