import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# --- 1. Load Data ---
# Reading from stations_full.csv as per the reference notebooks
df = pd.read_csv('Datasets/stations_full.csv', skiprows=[1])

# --- 2. Preprocessing ---
for col in df.columns:
    if col not in ['station_code', 'monitoring_location', 'state_name']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

target_features = ['do_max', 'bod_max', 'fecal_coliform_max']
df.dropna(subset=target_features, inplace=True)

# --- 3. Feature Engineering: Create Target Variable (WQI Category) ---
def classify_wqi(row):
    do_max = row['do_max']
    bod_max = row['bod_max']
    fecal_coliform_max = row['fecal_coliform_max']

    if do_max > 6.5 and bod_max < 2 and fecal_coliform_max < 100:
        return 'Good'
    elif 5 < do_max <= 6.5 and 2 <= bod_max < 5 and 100 <= fecal_coliform_max < 1000:
        return 'Moderate'
    else:
        return 'Poor'

df['wqi_category'] = df.apply(classify_wqi, axis=1)

# --- 4. Prepare Features (X) and Target (y) ---
if df['wqi_category'].nunique() < 2:
    print("The created target variable 'wqi_category' has only one class.")
    print("Cannot train a classifier. Please adjust the classification logic in classify_wqi function.")
    exit()

# Features as per ganga-deci.ipynb and others
features = [col for col in df.columns if '_min' in col or '_max' in col] + ['state_name']
X = df[features].copy()
y = df['wqi_category']

# Fill missing values
for col in X.select_dtypes(include=['number']).columns:
    X[col].fillna(X[col].mean(), inplace=True)

# Label Encoding for state_name
if 'state_name' in X.columns:
    le = LabelEncoder()
    X['state_name'] = le.fit_transform(X['state_name'].astype(str))
    # Note: Ideally we should save this encoder too if it's not handled elsewhere, 
    # but the notebooks don't explicitly save the state_encoder in the snippet provided (except implied).
    # We'll rely on the ensemble script's loading or save it if needed. 
    # For now, we follow the pattern of saving the model and scaler.

# --- 5. Feature Scaling ---
# SVM requires scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Using probability=True to support ROC curves in ensemble_model.py
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# --- 7. Evaluate the Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy}")

# --- 8. Save the Model ---
joblib.dump(model, 'models/svm_model.joblib')
print("Saved models/svm_model.joblib")

# Saving scaler as it's required for the ensemble script to transform new data correctly
# and ensures consistency with the training data used here.
# joblib.dump(scaler, 'models/scaler.joblib')
# print("Saved models/scaler.joblib")