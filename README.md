# 🏨 Hotel Booking Cancellation Prediction

## 📌 Project Overview
This project predicts hotel booking cancellations using a real-world dataset of ~119,000 reservations. It follows a full data science pipeline: data exploration, cleaning, feature engineering, encoding, and model comparison to identify the best classifier for predicting whether a booking will be canceled.

## 📊 Dataset
- **Records**: ~119,000 hotel bookings
- **Target**: `is_canceled` (binary: 1 = canceled, 0 = not canceled)
- **Features**: booking details, guest information, stay dates, room type, deposit type, distribution channel, and more

## 🔍 Project Workflow

### 1. Data Exploration
- Basic dataset information and summary statistics for numerical and categorical variables
- Identification of high-cardinality features (`country`, `agent`, `company`, `name`)
- Detection of noisy data (negative prices, zero-adult bookings, invalid children/babies counts)

### 2. Data Preprocessing
- Removal of features directly related to the target (to avoid data leakage)
- Handling of missing values
- Fixing noisy/invalid data points
- Feature engineering: new `number_of_bookings` feature representing each guest's booking history
- Encoding of categorical variables:
  - One-hot encoding for low-cardinality features
  - Frequency encoding for `country` (keeps geographic signal without exploding dimensionality)
- Train/test split (80/20, stratified) performed **before** scaling, to prevent data leakage

### 3. Model Building & Tuning
Five classification models are trained and compared, each tuned with `GridSearchCV` and stratified cross-validation:

| Model | Notes |
|-------|-------|
| Decision Tree | Baseline tree-based model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting, tuned with `scale_pos_weight` |
| Logistic Regression | Linear baseline, trained on scaled features |
| Gradient Boosting | Sequential boosting ensemble |

### 4. Model Evaluation
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC**
- **Confusion Matrix**

## 🛠️ Technologies Used
Python 3.x with Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and XGBoost.

## 📈 Results Summary

Performance on the test set (class 1 = canceled bookings):

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Decision Tree | 83.44% | 74.65% | 83.79% | 78.96% | 91.60% |
| Random Forest | 86.22% | 80.15% | 83.52% | 81.80% | 93.95% |
| **XGBoost** | **87.25%** | 81.08% | 85.58% | **83.27%** | **94.88%** |
| Logistic Regression | 78.36% | 68.10% | 78.31% | 72.85% | 87.84% |
| Gradient Boosting | 86.26% | 85.09% | 76.32% | 80.47% | 93.68% |

**🏆 Best model: XGBoost**, with the highest F1-score (83.27%) and AUC (94.88%), offering the best balance between correctly identifying cancellations (recall) and avoiding false alarms (precision).

Gradient Boosting stands out for the highest precision (85.09%), making it a strong candidate if minimizing false positives matters most, while XGBoost remains the most balanced choice overall.

## 🚀 How to Run
```bash
git clone https://github.com/billydassi101-ship-it/Hotel-Booking-Cancellation-Prediction.git
cd Hotel-Booking-Cancellation-Prediction
pip install -r requirements.txt
jupyter notebook "hotel-booking-cancellation-prediction.ipynb"
```

## 📁 Repository Structure
```
Hotel-Booking-Cancellation-Prediction/
├── hotel-booking-cancellation-prediction.ipynb   # Main notebook
├── README.md                                      # Project documentation
├── requirements.txt                               # Python dependencies
└── Hotel_Booking.csv                              # Dataset
```

## 📦 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
This project is open-source and available under the MIT License.

Made with ❤️ by Samuel DASSI
