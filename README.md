# Currency Recognition System Using Image Processing

A computer vision-based application designed to automatically detect, identify, and verify paper currency denominations using digital image processing techniques. This system is engineered to provide an efficient, automated alternative to manual currency verification, aiding visually impaired individuals and automating retail cash points.

---

## 🚀 Key Features

* **Preprocessing Pipeline:** Automatic image resizing, noise reduction (Gaussian/Median Filtering), and grayscale conversion.
* **Feature Extraction:** Utilizes advanced edge detection (Canny) and keypoint descriptors (SIFT/ORB) to extract unique currency design patterns, security threads, and geometrical layouts.
* **High-Accuracy Matching:** Compares extracted signatures against a database of authentic banknote templates using robust feature matching algorithm variants (FLANN / Brute-Force Matcher).
* **Real-time Recognition:** Designed to process and classify input frames via standard webcam feeds or static image uploads.
* **Intuitive Feedback:** Visual bounding boxes with overlay labels displaying the identified denomination alongside confidence parameters.

---

## 🛠️ Built With

* **Python 3.x** - Core programming environment.
* **OpenCV** - Primary library for image manipulation, feature extraction, and computer vision routines.
* **NumPy** - High-performance multi-dimensional array operations for processing image matrices.
* **Matplotlib** - Used for rendering analysis graphs, feature match visuals, and debugging pipelines.
* **Tkinter / PyQt** *(Optional)* - Lightweight Desktop GUI implementation.

---

## 📁 Project Structure
pip install -r requirements.txt
```text
Currency-Recognition-System-Using-Image-Processing/
│
├── dataset/                  # Contains sample source images of various denominations
│   ├── training/             # Golden master templates for database comparison
│   └── testing/              # Validation inputs for accuracy testing
│
├── modules/                  # Modular Python scripts
│   ├── pre_processing.py     # Resizing, grayscale, and filtering pipelines
│   ├── feature_extractor.py  # SIFT/ORB keypoint and edge detectors
│   └── matcher.py            # Feature matching and logic thresholds
│
├── app.py                    # Main application entry point (GUI/Webcam Controller)
├── requirements.txt          # Python package dependency manifest
└── README.md                 # Project

git clone [https://github.com/YOUR_USERNAME/Currency-Recognition-System-Using-Image-Processing.git](https://github.com/YOUR_USERNAME/Currency-Recognition-System-Using-Image-Processing.git)
cd Currency-Recognition-System-Using-Image-Processing

python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

pip install -r requirements.txt

python app.py

