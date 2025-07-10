#🍷 Wine Quality Prediction
This project uses machine learning techniques to predict the quality of wine based on physicochemical features such as acidity, sugar, pH, and alcohol content. The model is trained on the Wine Quality dataset and evaluates the quality on a scale from 0 to 10.

📌 Project Overview
Goal: Predict wine quality based on physicochemical features using machine learning.

Dataset: Wine Quality Data Set (UCI Repository)

Tech Stack: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

📊 Features Used:
Fixed Acidity

Volatile Acidity

Citric Acid

Residual Sugar

Chlorides

Free Sulfur Dioxide

Total Sulfur Dioxide

Density

pH

Sulphates

Alcohol

🔧 Models Implemented:
Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Each model's accuracy is compared, and the best-performing one is chosen for prediction.

🧪 Evaluation Metrics:
Accuracy Score

Confusion Matrix

Classification Report

📁 File Structure:
wine_quality_prediction/
│
├── wine_quality_prediction.ipynb   # Main notebook with data processing, EDA, training and evaluation
├── README.md                       # Project overview and documentation
🚀 How to Run
Clone the repository or download the .ipynb file.

Install the required dependencies:

```bash

pip install pandas numpy matplotlib seaborn scikit-learn
```
Open the notebook:
```
jupyter notebook wine_quality_prediction.ipynb
```
Run all cells to preprocess data, train models, and see the results.

📌 Results
The Random Forest Classifier achieved the highest accuracy (~[X]%) on the test set.

Alcohol and sulphates are among the most important features in predicting wine quality.

(Replace [X]% with actual accuracy if you want that added here, Master.)

🧠 Future Improvements
Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)

Ensemble learning

Web application using Flask or Streamlit

Integration with real-time wine quality scanning system

👨‍💻 Author
Safal Swayam
Master's in Computer Application, KIIT
Certified in Data Analytics, Machine Learning & AI
