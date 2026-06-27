import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

samples = pd.DataFrame([
    {
        'location': 'Jaipur',
        'Carpet Area': 500,
        'Super Area': 600,
        'Bathroom': 1,
        'Balcony': 0,
        'Furnishing': 'Unfurnished',
        'Transaction': 'Resale',
        'Status': 'Ready to Move',
        'Ownership': 'Freehold',
        'Car Parking': 0
    },
    {
        'location': 'Jaipur',
        'Carpet Area': 4000,
        'Super Area': 4500,
        'Bathroom': 5,
        'Balcony': 4,
        'Furnishing': 'Furnished',
        'Transaction': 'New Property',
        'Status': 'Ready to Move',
        'Ownership': 'Freehold',
        'Car Parking': 3
    }
])

predictions = model.predict(samples)

for i, price in enumerate(predictions, start=1):
    print(f"House {i} Predicted Price: ₹{round(price):,}")
    print(f"In Lakhs: {round(price/100000, 2)} Lakhs\n")