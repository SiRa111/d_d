import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model():
    data = pd.read_csv('feature_engineered_data.csv')

    # Define features (factors influencing priority)
    X = data[['Education Numeric', 'Income', 'Discrimination Score']]

    # Handle missing values in the features (fill NaN with median or 0)
    X['Education Numeric'] = X['Education Numeric'].fillna(X['Education Numeric'].median())
    X['Income'] = X['Income'].fillna(X['Income'].median())
    X['Discrimination Score'] = X['Discrimination Score'].fillna(X['Discrimination Score'].median())

    # Create a mock 'Priority Score' based on a weighted combination of the features
    data['Priority Score'] = (X['Education Numeric'] * 0.3 + 
                              (1 / (X['Income'] + 1e-6)) * 0.4 +  # Add small value to avoid division by zero
                              X['Discrimination Score'] * 0.3)

    y = data['Priority Score']

    # Handle missing values in the target (fill NaN with median or remove rows)
    y = y.fillna(y.median())

    # Split the data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'model/random_forest_priority_model.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

    # Use the model to rank individuals by priority
    data['Predicted Priority'] = model.predict(X)
    data.sort_values(by='Predicted Priority', ascending=False, inplace=True)

    # Save the ranked data
    data.to_csv('priority_ranked_data.csv', index=False)
    print("Priority ranking saved to 'priority_ranked_data.csv'")

if __name__ == "__main__":
    train_model()
