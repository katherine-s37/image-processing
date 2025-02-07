**Automatic Segmentation and Recognition of Individual Oracle Bone Script Characters from Raw Oracle Bone Images Using an Improved YOLOv8 Mode**

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

