# 💍 Render a Ring on Your Hand in an Image


Follow these steps to render a realistic 3D ring on a photo of your hand using depth maps and a 3D model.



---



## 🧤 Step 1: Take a Photo


- Capture a **clear photo of your hand** .

- Make sure the lighting is good and your fingers are visible.



---



## 🌊 Step 2: Generate a Depth Map

To create a 3D effect, you'll need a **depth map**  of your image.

- Use a free online tool like this:

👉 [Artificial Studio - Depth Map Generator](https://app.artificialstudio.ai/tools/image-depth-map-generator)

- Upload your photo and download the resulting depth image (usually a `.png`).



---



## 🛠️ Step 3: Set Up the project


**Clone the repository** :


```bash
git clone https://github.com/DenkoProg/VirtualRingTryOn
cd VirtualRingTryOn
```

**Install the dependencies** :


```bash
pip install -r requirements.txt
```


---



## 🚀 Step 4: Render the Ring


Run the following command in your terminal:



```bash
python src/cli.py render-ring-on-image \
    --rgb-path path/to/your/photo.jpg \
    --depth-path path/to/depth-map.png \
    --ring-model-path data/models/ring/ring.glb \
    --output-dir output/results
```


### Replace:


- `--rgb-path` with the path to your hand photo

- `--depth-path` with the depth image you downloaded

- `--output-dir` with the folder where you want the result saved



---



## ✅ Done!