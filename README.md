Brain Tumor Detection Web App (Deep Learning)

This project detects brain tumors in brain MRI images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The web app classifies uploaded MRI images into four categories:
•	Glioma
•	Meningioma
•	Pituitary
•	No Tumor

It provides prediction confidence, batch upload functionality, downloadable PDF reports, and visualizes model results in a user-friendly interface.
________________________________________

Requirements
•	Python 3.10+
•	TensorFlow
•	NumPy
•	OpenCV
•	Matplotlib
•	scikit-learn
•	Flask

Install dependencies:
pip install -r requirements.txt

Usage

1. Run the Web App
python app.py

•	Open your browser at http://127.0.0.1:5000/.
•	Upload single or multiple MRI images.
•	View predictions with confidence scores.
•	Download a PDF report summarizing results.

2. View Documents
•	Access the Documents page in the app to view/download:
o	Project Synopsis (PDF)
o	Design diagrams (PNG)
________________________________________

Model Details
•	Input: 224x224 RGB brain MRI images (preprocessed)
•	Architecture: Transfer learning using MobileNetV2 with custom Dense layers
•	Output: 4 neurons (softmax for multi-class classification)
•	Loss: categorical_crossentropy
•	Optimizer: Adam
•	Additional Features: Dropout for regularization, batch prediction support
________________________________________

Notes
•	This system is for academic, research, or training purposes only. It is not a medical diagnostic tool.
•	Images should be brain MRI scans; other image types may produce unreliable predictions.
•	Predictions are displayed with confidence percentages for transparency.
________________________________________

Author
Musab Salmani