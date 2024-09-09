import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering():
    # Load the combined preprocessed data
    data = pd.read_csv('combined_data.csv')

    # Binning and scaling education
    data['Education Bin'] = pd.cut(data['Qualification'],
                                  bins=[0, 2, 4, 5],
                                  labels=['Low Education', 'Medium Education', 'High Education'])
    data['Education Numeric'] = data['Education Bin'].map({
        'Low Education': 1,
        'Medium Education': 2,
        'High Education': 3
    })

    # Feature scaling for Education Numeric
    scaler = StandardScaler()
    data[['Education Numeric']] = scaler.fit_transform(data[['Education Numeric']])

    # Optional: Normalize other relevant columns if needed (example)
    # data[['Monthly Income']] = scaler.fit_transform(data[['Monthly Income']])

    # Save the feature-engineered data
    data.to_csv('feature_engineered_data.csv', index=False)

if __name__ == "__main__":
    feature_engineering()
