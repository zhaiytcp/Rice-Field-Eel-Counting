import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('.yaml')
    model.train(data='/root/code/dataset/dataset_visdrone/data.yaml',
                 cache=False,
                imgsz=640,
                epochs=300,
                batch=24,          
                workers=8,
                optimizer='SGD',
                lr0=0.01,          
                project='runs/train',
                name='exp',
                )