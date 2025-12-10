# yolo_trainer.py

from ultralytics import YOLO

class YoloTrainer:
    def __init__(self, model_name='yolov8n-obb.pt', data_yaml='data.yaml', epochs=50, imgsz=640, batch=16, project='yolo_runs'):
        """
        Initialize the YoloTrainer for YOLOv8 OBB.

        Args:
            model_name (str): Path to the pre-trained YOLO OBB model or model alias (e.g., 'yolov8n-obb.pt')
                              For rotated bounding box detection, ensure this is an OBB-specific model.
            data_yaml (str): Path to the dataset YAML file. Annotations should be in YOLO OBB format.
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
        
        # Load an OBB-specific model. The 'obb' suffix is crucial for rotated bounding boxes.
        self.model = YOLO(model_name)

    def train(self):
        """
        Train the YOLO OBB model on the specified dataset.
        """
        print(f"Starting OBB training with model: {self.model_name}")
        self.model.train(data=self.data_yaml,
                         epochs=self.epochs,
                         imgsz=self.imgsz,
                         batch=self.batch,
                         project=self.project)
        print("OBB training complete.")

    def validate(self):
        """
        Run validation on the trained OBB model.
        """
        print("Running OBB validation...")
        self.model.val() # The .val() method automatically adapts to the OBB model type
        print("OBB validation complete.")

    def predict(self, image_path):
        """
        Run prediction on a single image using the OBB model.

        Args:
            image_path (str): Path to the image file
        """
        print(f"Running OBB inference on {image_path}")
        # The .predict() method automatically handles OBB output when an OBB model is used
        results = self.model(image_path)
        
        # Display or save results. The 'show()' method will visualize rotated boxes.
        if results and len(results) > 0:
            results[0].show()  # Displays the image with OBB detections
            # If you want to save the image with predictions:
            # results[0].save(filename='predicted_image_with_obb.jpg')
            # You can also access OBB specific attributes from results if needed
            # For example, to print the OBB coordinates:
            # for r in results:
            #     if hasattr(r, 'obb'):
            #         for *xyxyxyxy, conf, cls in r.obb.data: # OBB data typically includes 8 coordinates for 4 corners
            #             print(f"Class: {int(cls)}, Confidence: {conf:.2f}, OBB Coords: {xyxyxyxy}")
        else:
            print("No detections found or issue with prediction.")
            
        return results
