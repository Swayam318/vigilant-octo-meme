# vigilant-octo-meme
CDC X Yhills OPEN PROJECTS 2025-2026 (Data Science PS)
# Satellite Imagery-Based Property Valuation 🛰️🏠

## 📌 Project Overview
This project implements a **Multimodal Machine Learning Pipeline** to predict property prices by combining traditional tabular data (e.g., bedrooms, square footage) with satellite imagery.

Standard real estate models often miss the "curb appeal" or neighborhood density factors that influence price. By processing satellite images using a **Deep Convolutional Neural Network (CNN)** and fusing this visual intelligence with a powerful **Ensemble Regressor**, this solution achieves superior predictive performance compared to tabular-only baselines.

**Key Achievement:**
- **Baseline (Tabular Only):** $R^2 \approx 0.86$
- **Final Model (Hybrid Ensemble):** $R^2 \approx 0.89$ (Significant improvement in precision)

---

## 📂 Repository Structure

```text
├── data_fetcher.py           # Script to download satellite images via API
├── preprocessing.ipynb       # Data cleaning, log-transformations, and scaling
├── model_training.ipynb      # Trains the CNN (ResNet18) to learn visual embeddings
├── hybrid_training.ipynb     # The MAIN pipeline: Feature Extraction -> Clustering -> Ensemble
├── explainability.ipynb      # Grad-CAM visualization to interpret model focus
├── train.csv                 # Historical property data
├── test.csv                  # Test dataset for submission
├── satellite_images/         # Folder containing images (named by ID)
├── enrollno_final.csv        # Final predictions (Submission file)
└── enrollno_report.pdf       # Detailed project report
