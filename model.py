import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv('Cleaned_data.csv')

# Prepare data
X = df[['total_sqft', 'bathroom', 'balcony', 'BHK']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
