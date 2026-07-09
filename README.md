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
- **Confusion Matrix**

## 🛠️ Technologies Used
Python 3.x with Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and XGBoost.

## 📈 Results Summary
Results are compared across all 5 models using a summary table and bar chart, highlighting the best-performing model based on F1-score for the canceled class.

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
