from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
house = pd.read_csv('dataset.csv')

# Preprocessing
le = LabelEncoder()
house['Neighborhood'] = le.fit_transform(house['Neighborhood'])

# Define features (X) and target variable (y)
X = house[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
y = house['Price']

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        square_feet = float(request.form['square_feet'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        neighborhood = request.form['neighborhood']
        year_built = int(request.form['year_built'])

        # Check if the label is present in the encoder's classes
        if neighborhood in le.classes_:
            # Transform the 'Neighborhood' label to numerical
            neighborhood_numerical = le.transform([neighborhood])[0]
        else:
            # Handle the case where the label is unseen
            neighborhood_numerical = -1  # Assign a default numerical value

        # Make predictions
        new_house = pd.DataFrame([[square_feet, bedrooms, bathrooms, neighborhood_numerical, year_built]],
                                  columns=['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt'])

        predicted_price = model.predict(new_house[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']])
        predicted_price = round(predicted_price[0], 2)

        return render_template('result.html', predicted_price=f'Rs {predicted_price:,.2f} Only -/')

if __name__ == '__main__':
    app.run(debug=True)
