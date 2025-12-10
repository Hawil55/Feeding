# yolo_trainer.py

from ultralytics import YOLO

class YoloTrainer:
    def __init__(self, model_name='yolov8n.pt', data_yaml='data.yaml', epochs=50, imgsz=640, batch=16, project='yolo_runs'):
        """
        Initialize the YoloTrainer.

        Args:
            model_name (str): Path to the pre-trained YOLO model or model alias (e.g., 'yolov8s.pt')
            data_yaml (str): Path to the dataset YAML file
            epochs (int): Number of training epochs
            imgsz (int): Image size (square)
            batch (int): Batch size
            project (str): Folder to save training results
        """
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.project = project
        self.model = YOLO(model_name)

    def train(self):
        """
        Train the YOLO model on the specified dataset.
        """
        print(f"Starting training with model: {self.model_name}")
        self.model.train(data=self.data_yaml,
                         epochs=self.epochs,
                         imgsz=self.imgsz,
                         batch=self.batch,
                         project=self.project,
                         verbose=True)
        print("Training complete.")

    def validate(self):
        """
        Run validation on the trained model.
        """
        print("Running validation...")
        self.model.val()
        print("Validation complete.")

    def predict(self, image_path):
        """
        Run prediction on a single image.

        Args:
            image_path (str): Path to the image file
        """
        print(f"Running inference on {image_path}")
        results = self.model(image_path)
        results[0].show()  # or results[0].save()
        return results
