# **Friday - Virtual Ring Try-On**
A **virtual ring try-on** system that detects hand landmarks, estimates 3D transformations, and overlays a 3D ring model on images and videos. This project utilizes **OpenCV, Open3D, Mediapipe, and Typer** for processing.

---

## **🚀 Features**
- ✅ **Hand Landmark Detection**: Uses **Mediapipe** to detect keypoints.
- ✅ **Depth Map Processing**: Loads depth logs for accurate 3D alignment.
- ✅ **3D Ring Overlay**: Uses **Open3D** to render a 3D ring model onto the hand.
- ✅ **Video Processing**: Supports **frame-by-frame rendering** and generates processed videos.
- ✅ **CLI Interface**: Uses **Typer** for command-line interaction.

---
## **⚙️ Installation**

### **1️⃣ Clone the Repository**
First, clone the repository from GitHub and navigate into the project directory:
```bash
git clone https://github.com/DenkoProg/VirtualRingTryOn.git
cd VirtualRingTryOn
```

### **2️⃣ Create a Virtual Environment**
It is recommended to create a virtual environment to manage dependencies.

#### **🔹 Create a Virtual Environment**
Run the following command:
```bash
python -m venv venv
source venv/bin/activate # For Linux/Mac
venv\Scripts\activate # For Windows
```

---

## **📦 Install Dependencies**
After activating the virtual environment, install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## **🎨 Render Ring Using CLI**
The **VirtualRingTryOn** project allows you to overlay a 3D ring on an image using a **command-line interface (CLI)**.

### **🖼️ Render a Ring on an Image**
Run the following command to overlay a 3D ring on an image:
```bash
python src/cli.py render-ring-on-image \
    --images-folder "data/images" \
    --landmarks-folder "data/results" \
    --rgb-path "data/images/original_1.png" \
    --depth-log-path "data/depth_logs/depth_logs_1.txt" \
    --ring-model-path "data/models/ring.glb"

