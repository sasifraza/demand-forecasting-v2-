import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os 

# File paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sales.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "model.pkl")

# Load Data 

def load_data():
    df= pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape}")
    print(df.head())
    return df
# wide format  -> Long format 

# Reshape wide to long

def reshape_data(df):
    df_long =pd.melt(
    df,
    id_vars=["id","item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="day",
    value_name="sales"
    )
    print(f"Reshaped:{df_long.shape}")
    return df_long

# Feature Engineering

def create_features(df):
    df=df.sort_values(["id","day"])

    # lag feature 
    df["lag_1"] = df.groupby("id")["sales"].shift(1)
    df["lag_7"] = df.groupby("id")["sales"].shift(7)
    df["lag_14"] =df.groupby("id")["sales"].shift(14)

    # rolling features 

    df["rolling_mean_7"] = df.groupby("id")["sales"].transform(
        lambda x:x.shift(1).rolling(7).mean()
    )

    df["rolling_mean_14"] = df.groupby("id")["sales"].transform(
        lambda x:x.shift(1).rolling(14).mean()
    )

    # Encode product ID
    df["id_encoded"] = df["id"].astype("category").cat.codes

    features = ["lag_1", "lag_7", "lag_14", 
            "rolling_mean_7", "rolling_mean_14",
            "id_encoded"]
    # keep only necessary columns     
    
    keep_cols = ["id", "day", "sales"] + features

    df=df[keep_cols]

    # drop NaN

    df=df.dropna()

    print(f"Feature created:{df.shape}")
    return df

def train_model(df):
    features=["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "id_encoded"]
    target="sales"

    X=df[features]
    y=df[target]

    # time split 80/20 r
    split= int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train,y_test  = y[:split], y[split:]

    # Train XGBoost
    model=xgb.XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 5,
        random_state = 42 
    )

    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))

    print(f"MAE, {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return model

def save_model(model):
    joblib.dump(model,MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    df = load_data()
    df = reshape_data(df)
    df = create_features(df)
    model = train_model(df)
    save_model(model)
    print("Training completes!")





    
    











