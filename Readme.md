# **🚨 DeepViolence Detector: A Real-time Video Violence Detection featuring Swin3D technology and easy to use GUI 🎥**

This repository presents a robust solution for real-time violence detection in video streams, leveraging a fine-tuned Swin Transformer V2 (Swin3D) model and an intuitive Graphical User Interface (GUI). The project covers the complete pipeline from data preprocessing and model training to local deployment and interactive inference.

## **📝 Table of Contents**

* [Project Overview](#bookmark=id.bohopgsin6ox)  
* [Features](#bookmark=id.q0mf4fp0uz2g)  
* [Applications](#bookmark=id.kx8ae761rarq)  
* [Dataset](#bookmark=id.f4xchyervvam)  
* [Model Architecture](#bookmark=id.c0ediltmwqd)  
* [Training](#bookmark=id.qyqltsblbo1e)  
* [Installation](#bookmark=id.slpg0b4gxehp)  
* [Usage](#bookmark=id.xitggurjyc10)  
* [Files in this Repository](#bookmark=id.7x3nj4q3hmz4)  
* [Results & Visuals](#bookmark=id.vxs30dfmhk0r)  
* [Contributing](#bookmark=id.23aelow8rai1)  
* [License](#bookmark=id.tn27kkrotwy8)  
* [Contact](#bookmark=id.hzg7ov2d6329)

## **🚀 Project Overview**

https://github.com/user-attachments/assets/4a3c158a-9bbb-4ebd-80a1-d0f7df7be75b

A demonstration showcasing our model's performance in detecting violence of a real life CCTV footage.

    The core mission of the DeepViolence Detector is to automatically identify violent acts within video content/ Live Stream as deployed. This capability is increasingly vital for enhancing public safety, content moderation, increasing crimes against women and overall security surveillance. By employing a powerful deep learning model and providing a user-friendly interface, this project aims to make advanced video analytics accessible.

The project encompasses:

* **Video Preprocessing**: Efficient extraction and transformation of video frames.  
* **Swin Transformer V2 (Swin3D) Model**: A state-of-the-art 3D convolutional neural network pre-trained on large video datasets, fine-tuned for binary classification (Violent/Non-Violent).  
* **Training Pipeline**: A comprehensive PyTorch-based setup for training and validating the model.  
* **Real-time GUI Application**: An interactive tkinter application that allows users to load video files and observe live predictions of violence with confidence scores.  
* **Local Deployment**: The application is designed to be packaged into a standalone executable for easy distribution and use without requiring a Python environment setup on the end-user's machine.

## **✨ Features**

* **Swin3D Backbone**: Utilizes torchvision.models.video.swin3d\_t, a powerful 3D Swin Transformer, capable of capturing both spatial and temporal features in video sequences.  
* **Transfer Learning**: The pre-trained Swin3D backbone is frozen, and only a custom classification head is trained, enabling efficient learning with less data. Selective unfreezing of later layers is also implemented for more fine-tuning.  
* **Robust Video Preprocessing**: Includes frame extraction, resizing, normalization, and data augmentation (horizontal flip, color jitter, rotation, Gaussian noise) for training.  
* **Interactive GUI (VD\_GUI.py)**:  
  * Browse and load local video files.  
  * Real-time video playback within the application.  
  * Live display of "VIOLENT" or "SAFE" predictions.  
  * Confidence meter indicating the model's certainty. 
  * The LIVE STREAM feature allows for real-time violence detection by the model through CCTV, drones, and similar surveillance technologies.  
  * Dynamic visual feedback (video border color changes based on prediction: Red for Violent, Green for Safe).  
  * Multi-threading for smooth video playback independent of prediction processing.  
* **Encapsulated Prediction Logic**: The ViolenceDetector class (video\_eval\_encapsulations.py) provides a clean API for loading the model and making predictions on video frames or sequences.  
* **Executable Packaging Ready**: Designed for easy conversion into a standalone .exe application using PyInstaller, facilitating deployment.

## **🌐 Applications**

The DeepViolence Detector can be applied in various real-world scenarios to enhance safety and content management:

* **Security Surveillance**: Monitoring public spaces, commercial establishments, or residential areas for suspicious or violent activities, enabling rapid response.  
* **Content Moderation**: Automatically flagging or filtering violent content on social media platforms, streaming services, or user-generated content sites to maintain community standards and protect users.  
* **Child Safety**: Identifying inappropriate or harmful video content for minors in educational or entertainment platforms.  
* **Sports Analytics**: Analyzing sports events for aggressive behavior or rule violations (e.g., in contact sports).  
* **Elderly Care Monitoring**: Detecting falls or unusual aggressive movements in assisted living facilities.  
* **Automated Incident Reporting**: Generating alerts or reports for security personnel when violent events are detected.

## **📊 Dataset**

This project utilizes the **"Real Life Violence Situations Dataset"** available on Kaggle.

**Dataset Structure:**

The dataset is expected to be organized into two main categories:

Real Life Violence Dataset/  
├── Violence/  
│   ├── V\_1.mp4  
│   ├── V\_2.mp4  
│   └── ...  
└── NonViolence/  
    ├── NV\_1.mp4  
    ├── NV\_2.mp4  
    └── ...

**Key Characteristics:**

* Consists of video clips categorized as either "Violence" (label 1\) or "NonViolence" (label 0).  
* Videos are processed by extracting a fixed number of evenly spaced frames (e.g., 16 frames per video) to capture temporal dynamics.  
* Each frame is resized to 224x224 pixels and normalized using ImageNet means and standard deviations, consistent with the pre-trained Swin3D model's requirements.

## **🧠 Model Architecture**

The core model for violence detection is defined in the VDC class (test\_video.py, video\_eval\_encapsulations.py), built upon torchvision.models.video.swin3d\_t.

**Model (VDC class):**

* **Backbone**: swin3d\_t(progress=True): This is a pre-trained Swin Transformer V2 for video classification. It processes video input as \[batch, channels, time, height, width\] (e.g., \[B, 3, 16, 224, 224\]).  
  * **Freezing**: The majority of the pre-trained backbone layers are initially frozen (param.requires\_grad=False) to leverage learned features and prevent catastrophic forgetting.  
  * **Selective Unfreezing**: Specifically, backbone.features\[6\] (a later stage of the Swin Transformer) and backbone.norm (final normalization layer) are unfrozen to allow for fine-tuning of higher-level features relevant to the specific task.  
* **Custom Head :** A small, trainable classification head is appended to the frozen/selectively unfrozen backbone. This head transforms the backbone's high-dimensional feature vector (768 dimensions) into a single output for binary classification.  
 
* **Output**: The model's forward method returns a single logit, which is then passed through a torch.sigmoid function during inference to get a probability between 0 and 1\.

## **🛠️ Installation**

To set up and run this project, ensure you have Python installed (preferably Python 3.8+). Using a virtual environment is highly recommended.

1. **Clone the repository:**  
   \# If this were a Git repository, you would clone it like this:  
   \# git clone \<repository\_url\>  
   \# cd deepviolence-detector

2. **Install dependencies:**  
   pip install torch torchvision numpy pandas opencv-python pillow tqdm torchmetrics kagglehub

   *Note: tkinter is usually included with Python installations.*  
3. Download the Dataset:  
   The project uses the "Real Life Violence Situations Dataset" from Kaggle. Y
4. Load and use our trained Model:  
   The VD\_GUI.py and test\_video.py scripts expect a pre-trained model named VD\_classification.pt. This model is generated and saved during the training process in violence-detection.ipynb. Ensure this file is present in the same directory as your application scripts or update the model\_path in ViolenceDetector initialization.

## **💡 Usage**

### **1\. Training the Model (violence-detection.ipynb)**

To train the violence detection model, execute the violence-detection.ipynb Jupyter notebook in an environment like Google Colab or Jupyter Lab.

The notebook will:

* Load and preprocess video data.  
* Initialize and train the VDC model.  
* Save the best-performing model as VD\_classification.pt.  
* Print training and validation metrics.

In this architecture, accuracy can reach 97%; however, we will update the model in the future to exceed the current limit and even include Zero Shot Learning capabilities.

### **2\. Running the GUI Application (VD\_GUI.py)**

Once you have the VD\_GUI.py script and the VD\_classification.pt model file in the same directory (along with video\_eval\_encapsulations.py), you can launch the interactive application:

python VD\_GUI.py

* **Instructions:**  
  1. The application will start with a "WAITING" status.  
  2. Click "Open Video" to select a video file (.mp4, .avi, etc.).  
  3. The video will start playing, and the model will continuously analyze frames for violence.  
  4. The "VIOLENCE DETECTION" panel will update with "VIOLENT" or "SAFE" predictions, a confidence percentage, and the video border will change color (Red for Violent, Green for Safe).  
  5. Click "Stop" to halt video processing.

### **3\. Running a Basic Video Inference Script (test\_video.py)**

This script provides a minimal example of how to load the model and make a prediction on a single video file.

* **Before Running:**  
  * Ensure VD\_classification.pt is accessible.  
  * Update the dir variable in test\_video.py to the path of your test video (e.g., dir="path/to/your/video.mp4").  
* **Execute:**  
  python test\_video.py

  The script will print "Violent" or "Non Violent" based on the prediction.

### **4\. Building the Executable (.exe)**

To package your GUI application into a standalone Windows executable, use **PyInstaller**.

**1\. Project Setup & Prerequisites:**

Ensure all necessary files are in a dedicated folder, e.g., ViolenceDetectorApp:

ViolenceDetectorApp/  
├── VD\_GUI.py  
├── video\_eval\_encapsulations.py  
├── VD\_classification.pt  
└── (any other assets like icons, if used)

**2\. Install PyInstaller:**

pip install pyinstaller

**3\. Generate the .spec File:**

Navigate to your project directory and run:

cd path/to/ViolenceDetectorApp  
pyinstaller \--onefile \--windowed VD\_GUI.py

* \--onefile: Creates a single executable.  
* \--windowed: Prevents console window.

**4\. Edit the .spec File (Crucial Step\!):**

Open VD\_GUI.spec in a text editor. You **must** explicitly include your model (.pt) and video\_eval\_encapsulations.py. You might also need to add hidden imports for torchvision.models.video or cv2.

Locate the a.datas list and modify it. Also, add relevant modules to hiddenimports.

**5\. Build the Executable:**

Save the modified .spec file, then run PyInstaller:

pyinstaller VD\_GUI.spec

**6\. Test Your Executable:**

Find DeepViolenceDetector.exe in the dist/ folder and test it thoroughly.

**7\. Troubleshooting:** Refer to common PyInstaller issues (missing files, hidden imports, etc.).

## **📂 Files in this Repository**

* violence-detection.ipynb: Jupyter notebook containing the full training pipeline, data loading, preprocessing, model definition, and training loop.  
* test\_video.py: A standalone Python script for performing basic inference on a single video file using the trained model.  
* video\_eval\_encapsulations.py: Defines the ViolenceDetector class, which encapsulates the model loading, preprocessing transforms, and prediction logic for video sequences. It also contains the VDC model architecture.  
* VD\_GUI.py: The Python script for the Graphical User Interface (GUI) application, enabling interactive video playback and real-time violence detection.  
* VD\_classification.pt (generated during training): The saved state dictionary of the trained VDC model.

## **🤝 Contributing**

Contributions are highly encouraged and welcome\! If you have suggestions for improvements, new features, or encounter any bugs, please consider:

1. **Fork the repository**.  
2. Create a new branch for your feature or bug fix:  
   git checkout \-b feature/YourFeatureName  
   \# or  
   git checkout \-b bugfix/FixDescription

3. Make your changes and ensure they adhere to the project's coding style.  
4. Commit your changes with a clear and concise message:  
   git commit \-m 'feat: Add Your New Feature'  
   \# or  
   git commit \-m 'fix: Resolve Bug Description'

5. Push your changes to your forked repository:  
   git push origin feature/YourFeatureName

6. Open a Pull Request to the main branch of this repository, describing your changes in detail.

## **📜 License**

This project is open-source and available under the MIT License, attached in the repo.

## **📧 Contact**

For any questions, feedback, or inquiries, please feel free to reach out:

* **Name**: Souradeep Dutta
* **Email**: aimldatasets22@gmail.com  
* **GitHub Profile**: [https://github.com/Souradeep2233](https://github.com/Souradeep2233)
