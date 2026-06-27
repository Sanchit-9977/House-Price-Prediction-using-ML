import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# LOAD DATA
df = pd.read_csv("data/house_prices.csv")
print("Loaded ✅")

# CLEAN PRICE
def convert_price(p):
    try:
        if isinstance(p, str):
            p = p.replace(',', '').strip()
            if 'Lac' in p:
                return float(p.replace('Lac', '')) * 100000
            elif 'Cr' in p:
                return float(p.replace('Cr', '')) * 10000000
        return float(p)
    except:
        return np.nan

if 'Amount(in rupees)' in df.columns:
    df['price'] = df['Amount(in rupees)'].apply(convert_price)
elif 'Price (in rupees)' in df.columns:
    df['price'] = df['Price (in rupees)'].apply(convert_price)
else:
    raise ValueError("Price column not found")

# CLEAN NUMERIC COLUMNS
def extract_number(x):
    try:
        x = str(x)
        num = ''.join(ch for ch in x if ch.isdigit() or ch == '.')
        return float(num) if num else np.nan
    except:
        return np.nan

numeric_like_cols = ['Carpet Area', 'Super Area', 'Bathroom', 'Balcony', 'Car Parking', 'Plot Area']

for col in numeric_like_cols:
    if col in df.columns:
        df[col] = df[col].apply(extract_number)

# SELECT FEATURES
selected_features = [
    'location',
    'Carpet Area',
    'Super Area',
    'Bathroom',
    'Balcony',
    'Furnishing',
    'Transaction',
    'Status',
    'Ownership',
    'Car Parking'
]

selected_features = [col for col in selected_features if col in df.columns]

df = df[selected_features + ['price']]
df = df.dropna(subset=['price'])

# REMOVE OUTLIERS
if 'Carpet Area' in df.columns:
    df = df[(df['Carpet Area'].isna()) | (df['Carpet Area'] < 10000)]

df = df[df['price'] < 1e8]

print("Final dataset shape:", df.shape)

# SPLIT
X = df.drop('price', axis=1)
y = df['price']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# PREPROCESSING
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=7, weights='distance')
}

results = {}
best_model = None
best_r2 = -999
best_name = ""

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODELS
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    score_percent = (1 - mape) * 100

    results[name] = {
        'MSE': mse,
        'R2': r2,
        'MAPE': mape,
        'Score': score_percent
    }

    print(f"\n{name}")
    print("MSE:", mse)
    print("R2 Score:", r2)
    print("MAPE:", mape)
    print("MAPE-based Score:", score_percent, "%")

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_name = name

# SAVE BEST MODEL
joblib.dump(best_model, "models/best_model.pkl")

print(f"\n🏆 Best Model: {best_name}")
print("Best R2 Score:", best_r2)
print("✅ Saved as models/best_model.pkl")