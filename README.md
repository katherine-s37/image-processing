## Automatic Segmentation and Recognition of Individual Oracle Bone Script Characters from Raw Oracle Bone Images Using an Improved YOLOv8 Mode


<img src="https://github.com/user-attachments/assets/c714a64e-2e39-446a-8ed4-26d5ed7d5eee" width="300" />
<img src="https://github.com/user-attachments/assets/fa57ced5-1c42-4a1c-b4cf-18894c3eb2ca" width="300" />
<img src="https://github.com/user-attachments/assets/3cf5591b-b301-4049-adb0-3f5004455be5" width="300" />




**Description:**

This project focuses on automating the segmentation and recognition of individual oracle bone script characters from raw oracle bone images using an improved YOLOv8 model. The project is divided into four main stages:

**1. Problem 1: Image Preprocessing and Feature Extraction**
 
Initially, several image preprocessing techniques were compared, including **Histogram Equalization, Adaptive Histogram Equalization, Homomorphic Filtering, Logarithmic Transformation, Power-law Transformation, and Grayscale Enhancement**. Homomorphic Filtering was selected as the most suitable preprocessing method for oracle bone images. Subsequently, three feature extraction techniques—**SIFT+BoW, HOG, and LBP**—were applied to extract key features from the images. Missing value handling, data normalization, and feature dimensionality reduction were performed to retain critical image features. Finally, the feature matrices from all three techniques were concatenated, resulting in a comprehensive feature matrix that fused visual information from three perspectives, creating a robust image preprocessing model.

**2. Problem 2: YOLOv8-Based Character Segmentation**
 
The YOLOv8 model was improved for automatic character segmentation in oracle bone images. The model was evaluated using **six-fold cross-validation** and metrics such as **loss**, **precision**, and **recall**. After analyzing the evaluation results, model parameters were further optimized to ensure excellent generalization performance. A fast and accurate oracle bone image segmentation model was established.
	
**3. Problem 3: Automatic Character Segmentation on 200 Oracle Bone Images**

Using the preprocessing model from Problem 1 and the segmentation model from Problem 2, 200 oracle bone images were automatically segmented into individual characters. The results were visualized and analyzed, with model parameters adjusted based on segmentation performance to achieve more precise results. The detailed segmentation results can be found in "Test_results.xlsx".
	
**4. Problem 4: Automatic Recognition of Segmented Characters**

The preprocessing model was applied to 50 test images. The improved YOLOv8 model was then used to segment the oracle bone images, resulting in **361 segmented characters**. The YOLOv8 initial model was used to retrain and fit the labeled oracle bone script data, and the newly trained model was applied for automatic text recognition of the segmented characters. Detailed recognition results are provided in the supplementary materials.

**Technologies:**
	• Machine Learning: YOLOv8, YOLOv5, SIFT+BoW, HOG, LBP
	• Image Processing: Homomorphic Filtering, Histogram Equalization, Grayscale Enhancement
	• Programming Languages: Python
	• Libraries & Tools: OpenCV, TensorFlow, PyTorch, Scikit-learn, NumPy, Matplotlib
 
**Skills:**
	• Image Preprocessing, Feature Extraction, Object Detection
	• Machine Learning Model Development, Model Optimization
	• Text Recognition, Data Annotation, Cross-validation
	• Problem Solving, Model Evaluation, Data Visualization
 

<img src="https://github.com/user-attachments/assets/ea93ff1a-cc91-43bf-bd3a-d9c27239df0d" width="300" />
<img src="https://github.com/user-attachments/assets/828bdf77-7856-4b0a-a2b8-984bac1154a1" width="90" />
<img src="https://github.com/user-attachments/assets/296e8fb1-f030-48a7-bc44-ecab1f5e1929" width="90" />
<img src="https://github.com/user-attachments/assets/d87e9424-3d79-4473-99b5-c4cede8790b1" width="90" />
<img src="https://github.com/user-attachments/assets/a8469568-bfab-4bbd-a143-a89f94ddf7a2" width="90" />
<img src="https://github.com/user-attachments/assets/cef10867-0a9e-42b1-b5ef-4b2076b4cb2a" width="90" />
<img src="https://github.com/user-attachments/assets/07cdd90f-178a-40c8-81a8-3fa0da27e0c5" width="90" />



