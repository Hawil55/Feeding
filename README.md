# Fish Feed Intake & Waste Tracker
A computer vision pipeline designed to quantify feeding behavior in aquaculture. This tool uses two YOLO models and ByteTrack to detect pellets, monitor "disappearance" events near fish mouths, and identify wasted feed based on spatial thresholds.

**🚀 Key Features**
**Dual-Model Detection**: Simultaneous tracking of feed pellets and "open mouth" events.
**Intelligent Logic: * Feeding Event:** Counted when a pellet disappears within a specific distance ($PROXIMITY\_THRESHOLD$) and time window of an open mouth detection.
**Waste Detection:** Counted when a pellet disappears below a defined vertical line ($WASTE\_THRESHOLD\_Y$) without a mouth detection.
**Priority Zones:** Supports instant pellet validation if feed enters through a specific "Drop Zone" at the top of the frame.
**Automated Reporting:** Generates an annotated video and a CSV log for every processed file.

🛠️ Installation & Setup
Clone the repository

Install dependencies: Bashpip install -r requirements.txt

Download models: in the /models folder:
pellet_best.onnx
mouth_best.onnx

⚙️ Configuration
You can adjust the tracking sensitivity in src/main.py

Pellet confidence threshold		    The minimum confidence for pellet detection to be consider valid
Minimum pellet detection frames		The minimum frames where a pellet must be detected before it is considered as a valid pellet where its disappearance can trigger a search for open mouth
Priority area ratio		            The ratio of the priority area on top of the video where all pellet detection becomes instantly valid without fulfilling the minimum pellet detection frames
Pellet disappearance frames		    The number of frames where the pellet can be not detected before it considers disappear
Open mouth confidence threshold		The minimum confidence for open mouth detection to be consider valid
Proximity threshold		            The search area (in pixel) for open mouth detection 
Open mouth detection window	    	The number of frames to look for an open mouth in case of a pellet disappearance
Feed waste ratio		              The ratio of the feed waste area on the bottom of the video where all pellet trackers that disappear there, will count as feed waste


📊 OutputThe script produces two main outputs in the /data/output folder:
Annotated Video: A visual replay showing tracking IDs, "Feeding Event" pop-ups, and the "Waste Line."
CSV Log: A detailed spreadsheet containing frame numbers, coordinates, and confidence scores for every feeding event.

🏗️ Technical Logic
The feeding detection follows a specific temporal sequence:
Tracking: Pellets are assigned a track_id.
Buffer: The system keeps a 50-frame "memory" of open mouth detections.
Validation: If a pellet disappears, the system looks back through the mouth history.Distance 
Formula: Proximity is calculated using the Euclidean distance formula:

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
