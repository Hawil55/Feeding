import ultralytics
from utils.YOLO_data_split import split_yolo_dataset
from utils.yolo_trainer import YoloTrainer
trainer = YoloTrainer(
    model_name='yolo11n-obb.pt',
    data_yaml='/home/ahawil/datasets/data.yaml',
    epochs=300,
    imgsz=1080,
    batch=64,
    project='data/Results2'
    
)

trainer.train()
trainer.validate()
# trainer.predict('test_images/sample.jpg')
