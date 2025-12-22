🚗 Car Price Prediction using Machine Learning

A complete end-to-end Machine Learning project that predicts the selling price of used cars based on various features such as year, fuel type, transmission, and kilometers driven.
The project includes data preprocessing, model training, evaluation, and deployment using Streamlit.

📌 Project Overview

Predicting the price of a used car is a common real-world regression problem.
In this project, multiple machine learning models are trained and evaluated, and the best-performing model is deployed as an interactive web application.

This project demonstrates:

Practical data preprocessing

Model comparison

Feature scaling

Model persistence

ML deployment

🧠 Machine Learning Models Used

Linear Regression

Decision Tree Regressor

Random Forest Regressor ✅ (Best Model)

Gradient Boosting Regressor

🏆 Best Model

Random Forest Regressor
Selected based on R² score and error metrics (MAE & RMSE).

🛠️ Tech Stack

Programming Language: Python

Libraries:

NumPy

Pandas

Scikit-learn

Joblib

Deployment: Streamlit

IDE: VS Code

📂 Project Structure
Car-Price-Prediction/
│
├── train.py                 # Model training & evaluation
├── app.py                   # Streamlit web application
├── car data.csv             # Dataset
├── best_car_price_model.pkl # Trained ML model
├── scaler.pkl               # Feature scaler
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

🔍 Dataset Description

The dataset contains information about used cars, including:

Manufacturing Year

Present Price

Kilometers Driven

Fuel Type

Seller Type

Transmission Type

Number of Previous Owners

Target variable:

Selling Price

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/Car-Price-Prediction.git
cd Car-Price-Prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

🧪 Train the Model

Run the training script to train models and save the best one:

python train.py


This will generate:

best_car_price_model.pkl

scaler.pkl

🚀 Run the Streamlit App
streamlit run app.py


Open the local URL shown in the terminal to access the web app.

🌐 Streamlit App Features

User-friendly UI

Real-time car price prediction

Consistent preprocessing with trained model

Production-ready ML inference

📊 Model Evaluation Metrics

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

These metrics ensure reliable and interpretable model performance.

💡 Key Learnings

Importance of matching training & deployment pipelines

Handling categorical variables correctly

Avoiding feature mismatch errors

Deploying ML models professionally

📈 Future Improvements

Hyperparameter tuning

Feature importance visualization

Model explainability (SHAP)

Cloud deployment (Streamlit Cloud / AWS)

👤 Author

Aadarsh Shukla
Aspiring Data Scientist & Machine Learning Engineer
Email: shuklaaadarsh00@gmail.com
Linkedin:https://www.linkedin.com/in/aadarsh-shukla-803147370
📌 Passionate about building real-world ML solutions

⭐ If you like this project

Give it a star ⭐ on GitHub — it motivates me to build more!
