# Eagle Eye: Automated Helmet Violation & OCR Pipeline

**Eagle Eye** is a fully automated, end-to-end computer vision pipeline tailored for smart-city traffic monitoring. It natively detects motorcycle riders bypassing helmet laws, mathematically isolates their license plates, enhances the visual crops, and extracts precise alphanumeric registration numbers using AI optical character recognition.

## Key Features

* **Multi-Stage Detection:** Combines standard YOLOv8 for object detection (identifying `riders` vs pedestrians) with a fine-tuned, dataset-purified custom PyTorch model for rigorous helmet validation ([Helmet](cci:2://file:///Users/tania/Desktop/eagle_eye/watcher.py:98:0-383:68) vs `No_Helmet`).
* **Cloud Plate Targeting:** Pings the targeted cropped motorcycle image securely to the Roboflow Inference API for instant localization of Indian license plate bounding regions.
* **Classical Preprocessing Engine:** Extracts the tiny plate area through a rigorous internal OpenCV engine:
  - `Geometric 2x Interpolation Scaling`
  - `Grayscale Normalization`
  - `CLAHE (Contrast Limited Adaptive Histogram Equalization)`
  - `Aggressive Unsharp Masking Filtering`
* **Hardware-Accelerated EasyOCR:** Uses Apple Silicon (MPS GPU) processing locally to rip alphanumeric text out of the enhanced images natively without cloud bottlenecks.
* **Dynamic Auditing:** Assembles real-time violation CSV logging tracking absolute timestamps, license plate strings, and standard violation code flags dynamically as frames pass through.

## How It Works

The architecture actively filters out noise and false-positives using programmatic **Overlap Boundary calculations**—making sure an identified "No_Helmet" hit intrinsically belongs directly on the motorcycle operator before executing an extract.

1. Images or frames are scanned by the Base Model (People/Bikes).
2. The logic strictly binds the rider area.
3. The Custom Model scans exclusively that rider for helmets.
4. If missing, the violator's motorcycle gets passed to Roboflow.
5. Plate coordinates are sent back to the script, which crops, mathematically enhances, pads, and feeds the resulting image into EasyOCR (`allowlist='A-Z0-9'`).
6. Bounding boxes are natively highlighted back onto the visual matrix, and violations are dropped into [violations_log.csv](cci:7://file:///Users/tania/Desktop/eagle_eye/runs/detect/violations_log.csv:0:0-0:0).

## Tech Stack
* **Python 3.10+**
* **Ultralytics / YOLOv8**
* **OpenCV / NumPy / Python Regex**
* **EasyOCR** (Integrated with Native MPS execution)
* **Roboflow Inference SDK**

## Usage
Launch the executable watcher and direct it at a specified media file (image/video).

```bash
python watcher.py --source "path/to/image_or_video.jpg"
