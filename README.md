# Hotel Booking Cancellation Prediction
### An End-to-End Machine Learning Pipeline with Clustering & Dimensionality Reduction

**Author:** Tursunai Turumbekova  
**Stack:** Python · scikit-learn · XGBoost · pandas · matplotlib · seaborn

---

## Project Overview

Hotel booking cancellations are a major source of revenue loss in the hospitality industry. This project builds and evaluates a **binary classification pipeline** to predict whether a hotel booking will be cancelled before arrival, enabling revenue managers to take proactive action through overbooking policies, early promotions, and targeted retention outreach.

The project extends beyond supervised learning to cover **unsupervised clustering** and **dimensionality reduction**, demonstrating a complete ML engineering workflow.

---

## Key Results

| Model | AUC | AP | Train (s) |
|---|---|---|---|
| Logistic Regression | 0.898 | 0.859 | 42.0 |
| K-Nearest Neighbors | 0.824 | 0.747 | 1.2 |
| **Random Forest (tuned)** | **0.957** | **0.939** | 36.5 |
| XGBoost | 0.952 | 0.930 | 4.2 |
| Gradient Boosting | 0.931 | 0.902 | 37.4 |

- Best model: **Random Forest** — AUC 0.957, AP 0.939
- Production candidate: **XGBoost** — near-identical AUC at 9× faster training
- At F1-optimal threshold (0.44): **84% recall, 87% precision**

---

## Project Structure

```
hotel_cancellation_ml_portfolio.ipynb   # Main notebook
submission_rf_final.csv                 # Test set predictions
README.md
```

### Notebook Sections

**Part 1 — Supervised Learning**
- Exploratory data analysis (ADR distribution, cancellation rate by lead time)
- Preprocessing: outlier removal, missing value imputation, stratified train/val split
- Model training and comparison across 5 classifiers
- ROC and Precision-Recall curves
- Permutation importance (top 20 features)
- Calibration curve analysis
- Confusion matrix at F1-optimal threshold
- Final model retrained on full dataset for test inference

**Part 2 — Unsupervised Clustering**
- K-Means with elbow method for k selection
- DBSCAN with parameter tuning
- Spectral Clustering
- Algorithm comparison across 5 synthetic datasets with varied geometry

**Part 3 — Dimensionality Reduction**
- PCA: cumulative explained variance analysis, 2D projection
- t-SNE: 2D embedding with perplexity tuning
- Side-by-side comparison of both methods

---

## Key Findings

- **`country_PRT`**, **`total_of_special_requests`**, and **`lead_time`** are the three strongest cancellation predictors
- **Deposit policy is the most actionable business lever** — non-refundable deposits strongly suppress cancellations
- Random Forest probabilities are S-curve distorted and require post-hoc calibration (Platt scaling or isotonic regression) before use in business logic
- No single clustering algorithm dominates across all data geometries — algorithm selection must be driven by the structural properties of the data
- PCA retains only 21.6% of variance in 2D; t-SNE produces clean digit cluster separation by preserving local neighbourhood structure

---

## Data

This project uses the [Hotel Booking Demand dataset](https://www.sciencedirect.com/article/pii/S2352340918315191) by Antonio, Almeida & Nunes (2019).

Download `hotel_bookings.csv` from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and update the data loading path in the notebook.

---

## Setup

```bash
git clone https://github.com/tursunait/hotel-cancellation-ml.git
cd hotel-cancellation-ml
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jinja2
```

Then open `hotel_cancellation_ml_portfolio.ipynb` in Jupyter and update the data path in the **Data Setup** cell.
