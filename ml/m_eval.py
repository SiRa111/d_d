import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model():
    data = pd.read_csv('data/processed/feature_engineered_data.csv')

    # Load the model
    model = joblib.load('model/random_forest_model.pkl')

    # Define features and target
    X = data[['Education Numeric', 'Income', 'Number of Vehicles', 'Discrimination Score']]
    y = data['Need for Caste Certificate']

    # Predict
    y_pred = model.predict(X)

    # Evaluate
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
