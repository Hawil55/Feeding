#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<THIS ONE WORKS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import ultralytics
from ultralytics import YOLO

# --- YoloTrainer Class Definition ---
class YoloTrainer:
    def __init__(self, model_name='yolov8n.yaml', data_yaml='data.yaml', epochs=50, imgsz=640, batch=16, project='yolo_runs'):
        """
        Initialize the YoloTrainer.
        """
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.project = project
        
        # 👈 FIX 1: Load the P2 architecture directly in the constructor.
        # This builds the Large model with the P2 head from scratch.
        self.model = YOLO(model_name) 
        
    def train(self):
        """
        Train the YOLO model on the specified dataset with custom hyperparameters.
        """
        print(f"Starting training from scratch with YOLOv8L P2 architecture: {self.model_name}")
        self.model.train(data=self.data_yaml,
                         epochs=self.epochs,
                         imgsz=self.imgsz,
                         batch=self.batch,
                         project=self.project,
                         verbose=True,
                         mosaic=0.0,    # Disables Mosaic augmentation for tiny objects.
                         box=10.0,       # Increases box loss weight for better localization.
                         
                         # ❌ The 'cfg' argument is REMOVED to avoid the SyntaxError.
                        )
        print("Training complete.")

    def validate(self):
        """
        Run validation on the trained model.
        """
        print("Running validation...")
        self.model.val()
        print("Validation complete.")

# --- Execution Code ---

trainer = YoloTrainer(
    # 👈 FIX 2: Use the model architecture YAML that includes P2 and the Large size (L).
    model_name='yolov8l-p2.yaml', 
    data_yaml='/home/ahawil/pellet_detector_model/data_splitted2/data.yaml',
    epochs=300,
    imgsz=1080,
    batch=16, 
    project='/home/ahawil/pellet_detector_model/results4'
)

# Start the training process
trainer.train()

# Run the validation process
trainer.validate()

# trainer.predict('test_images/sample.jpg')
