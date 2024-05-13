import cv2
import threading
#import cProfile as profile
from queue import Queue
from PyFaceDet import facedetectcnn
from multiprocessing import cpu_count

class FrameReader(threading.Thread):
    def __init__(self, video_source=0, frame_queue=None, stop_event=None):
        super().__init__()
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.video_capture = None  # 初始化时不直接打开视频源

    def open_video_source(self):
        self.video_capture = cv2.VideoCapture(self.video_source) #在此处开始获取默认分辨率与帧率
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print(f"Camera resolution: {self.width}x{self.height}, FPS: {fps}")

    def run(self):
        if not self.open_video_source():
            print("Failed to open video source.")
            self.stop_event.set()
            return
        
        while not self.stop_event.is_set():
            ret, frame = self.video_capture.read()
            if not ret:
                print("Frame read failed.")
                self.stop_event.set()
                break
            self.frame_queue.put(frame)
        self.video_capture.release()

class FaceDetector(threading.Thread):
    def __init__(self, frame_queue, result_queue, stop_event):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
    
    def run(self):
        while not self.stop_event.is_set():
            frame = self.frame_queue.get()
            if frame is None:
                continue
            faces = facedetectcnn.facedetect_cnn(frame)
            self.result_queue.put((frame, faces))
        self.frame_queue.task_done()

class VideoOutputAndDisplay(threading.Thread):
    def __init__(self, result_queue, output_path, stop_event):
        threading.Thread.__init__(self)
        self.result_queue = result_queue
        self.output_path = output_path
        self.stop_event = stop_event
        self.writer = None
        self.width = None
        self.height = None
        self.frame_count = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        while not self.stop_event.is_set():
            frame, faces = self.result_queue.get()
            if frame is None:
                continue
            
            if self.writer is None:
                self.width = frame.shape[1]
                self.height = frame.shape[0]
                #self.writer = cv2.VideoWriter(self.output_path, fourcc, 25.0, (4320, 2430))
                self.writer = cv2.VideoWriter(self.output_path, fourcc, 25.0, (self.width, self.height))
            
            for face in faces:
                x, y, L, W,confidence,angel = face
                cv2.rectangle(frame, (x, y), (x+L, y+W), (255, 0, 0), 2)
            
            # 更新并显示处理帧数
            self.frame_count += 1
            cv2.putText(frame, f"Frames Processed: {self.frame_count}", (10, self.height - 30), 
                       self.font, 1, (0, 255, 0), 2)
            
            # 更新并显示检测到的人脸数
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 30), 
                       self.font, 1, (0, 0, 255), 2)
            
            # 写入视频帧
            self.writer.write(frame)
            
            # 显示处理后的帧
            display_frame = cv2.resize(frame, (self.width, self.height))
            cv2.imshow('Processed Frame (Reduced Size)', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                #profile1.disable()
                #profile1.print_stats()
            
            self.result_queue.task_done()
        
        if self.writer is not None:
            self.writer.release()
def create_threads(num_detectors, video_path, output_path, stop_event):
    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)
    
    frame_reader = FrameReader(video_path, frame_queue, stop_event)
    
    detectors = [FaceDetector(frame_queue, result_queue, stop_event) for _ in range(num_detectors)]
    
    video_output_and_display = VideoOutputAndDisplay(result_queue, output_path, stop_event)
    
    return frame_reader, detectors, video_output_and_display

def start_and_join_threads(frame_reader, detectors, video_output_and_display):
    frame_reader.start()
    for detector in detectors:
        detector.start()
    video_output_and_display.start()
    
    frame_reader.join()
    for detector in detectors:
        detector.join()
    video_output_and_display.join()
if __name__ == "__main__":
    #profile1 = profile.Profile()
    #profile1.enable()
    video_path = 1   #若该项为数字，则为摄像头输入
    output_path = "output_path.mkv"
    num_threads = cpu_count()-1  # 控制FaceDetector线程数以及假设VideoOutputAndDisplay作为一个单独线程，建议等于CPU核心数-1
    
    stop_event = threading.Event()
    frame_reader, detectors, video_output_and_display = create_threads(num_threads-1, video_path, output_path, stop_event)
    
    start_and_join_threads(frame_reader, detectors, video_output_and_display)
    
    #profile1.disable()
    #profile1.print_stats()
    print("视频处理完成，已保存为", output_path)
