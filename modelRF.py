import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load and preprocess the dataset
def load_and_preprocess_data():
    data = pd.read_csv(r'NNN.csv')
    # Drop unnecessary columns
    data = data[['DISTRICT', 'Year', 'Actual(mm)']].dropna()
    # Encode DISTRICT column (convert to numerical codes)
    data['DISTRICT'] = data['DISTRICT'].astype('category').cat.codes
    # Define input features (X) and target variable (y)
    X = data[['DISTRICT', 'Year']]  # Features
    y = data['Actual(mm)']           # Target
    return X, y

# Train the RandomForest model and save it as a pickle file
def train_and_save_model():
    X, y = load_and_preprocess_data()
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model trained. MSE: {mse}, R² score: {r2}")
    # Save the trained model to a pickle file
    with open('rainfall_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

if __name__ == '__main__':
    train_and_save_model()
