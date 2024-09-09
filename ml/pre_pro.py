import pandas as pd

def load_data():
    # Load data from CSV files
    education = pd.read_csv(r'C:\Users\Administrator\Desktop\dd\data\education_data.csv')
    economic = pd.read_csv(r'C:\Users\Administrator\Desktop\dd\data\economic_data.csv', encoding='utf-8')
    social = pd.read_csv(r'C:\Users\Administrator\Desktop\dd\data\social_data.csv')
    return education, economic, social

def preprocess_data(education, economic, social):
    # Preprocess education data
    education['Qualification'] = education['Qualification'].map({
        'Below 10th': 1,
        '10th': 1,
        '12th': 2,
        'Undergraduate': 3,
        'Postgraduate': 4,
        'Ph.D.': 5
    })

    # Preprocess economic data
    economic['Monthly Income'] = economic['Monthly Income'].str.replace('₹', '', regex=False)  # Remove ₹
    economic['Monthly Income'] = economic['Monthly Income'].str.replace(',', '', regex=False)  # Remove ,
    economic['Monthly Income'] = economic['Monthly Income'].str.replace('?', '', regex=False)  # Remove ?
    economic['Monthly Income'] = economic['Monthly Income'].str.replace('-', '', regex=False)  # Remove -
    
    # Convert to numeric, setting errors to NaN
    economic['Monthly Income'] = pd.to_numeric(economic['Monthly Income'], errors='coerce')
    
    # Fill NaN values with 0 and convert to int
    economic['Monthly Income'] = economic['Monthly Income'].fillna(0).astype(int)

    # Define income brackets and labels
    brackets = [0, 10000, 20000, 50000, float('inf')]
    labels = ['Very Low', 'Low', 'Medium', 'High']

    # Assign income brackets
    economic['Income Bracket'] = pd.cut(economic['Monthly Income'], bins=brackets, labels=labels)

    # Define and apply the function to calculate the discrimination score
    def map_discrimination_score(row):
        responses = [
            row['Neighborhood Discrimination'],
            row['School/Office Discrimination'],
            row['Public Places Discrimination'],
            row['Religious Places Discrimination'],
            row['Hospital Discrimination']
        ]
        
        # Convert to integer, using 0 for non-numeric or empty values
        valid_responses = []
        for response in responses:
            try:
                value = int(response)
                valid_responses.append(value)
            except (ValueError, TypeError):
                valid_responses.append(0)  # or you can choose to omit this value

        # Calculate the average score
        if valid_responses:
            return sum(valid_responses) / len(valid_responses)
        else:
            return None

    # Apply the function to each row
    social['Discrimination Score'] = social.apply(map_discrimination_score, axis=1)

    # Combine data
    combined_data = pd.merge(education, economic, on='ID')
    combined_data = pd.merge(combined_data, social, on='ID')

    # Save the processed data
    combined_data.to_csv(r'combined_data.csv', index=False)

if __name__ == "__main__":
    education, economic, social = load_data()
    preprocess_data(education, economic, social)
