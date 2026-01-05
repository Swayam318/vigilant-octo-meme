# vigilant-octo-meme
CDC X Yhills OPEN PROJECTS 2025-2026 (Data Science PS)
# Satellite Imagery-Based Property Valuation 🛰️🏠

## 📌 Project Overview
This project implements a **Multimodal Machine Learning Pipeline** to predict property prices by combining traditional tabular data (e.g., bedrooms, square footage) with satellite imagery.

Standard real estate models often miss the "curb appeal" or neighborhood density factors that influence price. By processing satellite images using a **Deep Convolutional Neural Network (CNN)** and fusing this visual intelligence with a powerful **Ensemble Regressor**, this solution achieves superior predictive performance compared to tabular-only baselines.

## 📂 Repository Structure

```text
├── data_fetcher.py           # Script to download satellite images via API
├── preprocessing.ipynb       # Data cleaning, log-transformations, and scaling
├── model_training.ipynb      # Trains the CNN (ResNet18) to learn visual embeddings
├── hybrid_training.ipynb     # The MAIN pipeline: Feature Extraction -> Clustering -> Ensemble
├── explainability.ipynb      # Grad-CAM visualization to interpret model focus
├── 22322030_final.csv        # Final predictions (Submission file)
└── 22322030_report.pdf       # Detailed project report

## 🛠️ Tech Stack
Deep Learning: PyTorch, Torchvision (ResNet18)
Machine Learning: Scikit-Learn, XGBoost, HistGradientBoosting
Data Manipulation: Pandas, NumPy
Image Processing: PIL, OpenCV
Visualization: Matplotlib, Seaborn
APIs: Google Maps Static API

## 🚀 How to Run the Pipeline
Follow these steps in order to reproduce the results.

1. Environment Setup
Install the required dependencies:
pip install pandas numpy torch torchvision scikit-learn xgboost opencv-python matplotlib tqdm requests

2. Data Acquisition (Optional)
If you do not have the images yet, use data_fetcher.py.
Prerequisite: You need a valid Google Maps Static API Key.

(i) Set your API Key:
Linux/Mac: export GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
Windows (CMD): set GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
Windows (PowerShell): $env:GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"

(ii) Run the Script:
python data_fetcher.py
Note: This script reads from data/train.csv and saves images to data/images/. Ensure your CSV files are in the expected folder or adjust the paths in the script.

3. Data Preparation: Run preprocessing.ipynb.
Input: train.csv, test.csv
Action: Performs log-transformations on skewed features (sqft_lot, price) and standard scaling. Maps image paths to IDs.
Output: processed_train.csv, processed_test.csv4.

4. Train the Visual Feature Extractor: Run model_training.ipynb.
Action: Fine-tunes a pretrained ResNet18 CNN on the house price regression task.
Output: Saves the trained model weights to multimodal_model_best.pth.

5. Train the Hybrid Ensemble (Final Model): Run hybrid_training.ipynb.
Action: 1. Loads the trained CNN and extracts a 64-dimensional visual embedding vector for every house.
        2. Performs Geospatial K-Means Clustering to create "Neighborhood" features from Lat/Long.
        3. Trains a Voting Regressor (Ensemble of XGBoost + Random Forest + HistGradientBoosting).
Output: Generates the final submission file 22322030_final.csv.

6. Explainability: Run explainability.ipynb.
Action: Uses Grad-CAM to visualize heatmaps over satellite images, showing which areas (e.g., green spaces, driveways) the model focused on.
Output: explainability_report.png

## 🧠 Model Architecture: The solution uses a Late Fusion architecture:

1. Visual Branch:
Input: 224x224 RGB Satellite Image.
Backbone: ResNet18 (Pretrained on ImageNet, fine-tuned).
Output: 64-dim feature vector representing visual characteristics.

2. Tabular Branch:
Input: 18 numerical features (Bedrooms, Grade, Yr Built, etc.).
Augmentation: Interaction terms (Density) + Spatial Clusters (K-Means).

3. Fusion & Prediction:
The visual embeddings are concatenated with processed tabular data.
Passed to a Voting Regressor ($w_1 \cdot XGB + w_2 \cdot HGB + w_3 \cdot RF$) for final price prediction.


