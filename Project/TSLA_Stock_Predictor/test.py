import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DTLearner import DTLearner
from RTLearner import RTLearner
from BagLearner import BagLearner

# ✅ Load Tesla stock data (Remove Date & Volume)
def load_data(file_path):
    """Load Tesla stock data, preprocess, and normalize."""
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, usecols=[1, 2, 3, 4, 5])  # Open, High, Low, Close, Adj Close

    # Extract Features (X) and Target (Y)
    X = data[:, :-1]  # Open, High, Low, Close
    Y = data[:, -1]   # Target: Adjusted Close price

    # 🔹 Store Mean & Std for Normalization
    global X_mean, X_std, Y_mean, Y_std
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    Y_mean, Y_std = np.mean(Y), np.std(Y)

    # Standardize Features & Target
    X = (X - X_mean) / X_std
    Y = (Y - Y_mean) / Y_std

    return X, Y, data  # Also return raw data for validation

# ✅ Evaluate Model Performance
def evaluate_model(learner, train_X, train_Y, test_X, test_Y, name):
    """Train & evaluate a learner, returning performance metrics."""
    learner.add_evidence(train_X, train_Y)
    pred_Y = learner.query(test_X)

    # Convert back to actual prices
    pred_Y_actual = pred_Y * Y_std + Y_mean
    test_Y_actual = test_Y * Y_std + Y_mean

    # Compute Metrics
    corr = np.corrcoef(pred_Y_actual, test_Y_actual)[0, 1]
    mae = mean_absolute_error(test_Y_actual, pred_Y_actual)
    rmse = np.sqrt(mean_squared_error(test_Y_actual, pred_Y_actual))
    r2 = r2_score(test_Y_actual, pred_Y_actual)

    print(f"\n📌 {name} Performance:")
    print(f"   ✅ Correlation: {corr:.4f}")
    print(f"   ✅ MAE: ${mae:,.2f}")
    print(f"   ✅ RMSE: ${rmse:,.2f}")
    print(f"   ✅ R² Score: {r2:.4f}")

    return corr, pred_Y_actual

# ✅ Predict Future Tesla Stock Price
def predict_future_stock_price(learner, X_raw):
    """Predict future stock price of TSLA for a user-specified year & month."""
    print("\n🔮 Predicting Future Stock Price...")

    # ✅ Get User Input
    year = int(input("\nEnter year (XXXX): "))
    month = int(input("Enter month (1-12): "))

    # ✅ Use Last Known Features
    last_known_features = X_raw[-1, :-1]  # Last row (Open, High, Low, Close)

    # ✅ Normalize Features
    normalized_features = (last_known_features - X_mean) / X_std

    # ✅ Predict & Convert Back
    predicted_price_normalized = learner.query(normalized_features.reshape(1, -1))[0]
    predicted_price_actual = predicted_price_normalized * Y_std + Y_mean

    print(f"\n📌 Predicted Tesla Adjusted Close Price for {year}-{month}: ${predicted_price_actual:.2f}")

# ✅ Main Execution Function
def main():
    """Run ML experiments on Tesla stock data & generate predictions."""
    file_path = "Data/TSLA.csv"
    X, Y, X_raw = load_data(file_path)

    # ✅ Split Data (60% Train, 40% Test)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.6 * len(X))
    train_X, test_X = X[indices[:train_size]], X[indices[train_size:]]
    train_Y, test_Y = Y[indices[:train_size]], Y[indices[train_size:]]

    print("\n🔍 Evaluating ML Models on Tesla Stock Data...")

    # ✅ Define ML Models
    learners = {
        "DTLearner": DTLearner(leaf_size=40, verbose=False),
        "RTLearner": RTLearner(leaf_size=40, verbose=False),
        "BagLearner (DT)": BagLearner(learner=DTLearner, kwargs={"leaf_size": 40}, bags=5, verbose=False),
        "BagLearner (RT)": BagLearner(learner=RTLearner, kwargs={"leaf_size": 40}, bags=5, verbose=False),
    }

    correlations = {}
    predictions_dict = {}

    for name, learner in learners.items():
        correlations[name], predictions_dict[name] = evaluate_model(learner, train_X, train_Y, test_X, test_Y, name)

    # ✅ Compare with Baseline: 5-Day Rolling Average
    rolling_avg_pred = pd.Series(test_Y * Y_std + Y_mean).rolling(window=5, min_periods=1).mean()
    rolling_corr = np.corrcoef(rolling_avg_pred, test_Y * Y_std + Y_mean)[0, 1]
    correlations["Rolling Avg (Baseline)"] = rolling_corr

    print(f"\n📌 Rolling Average Baseline Performance:\n   ✅ Correlation: {rolling_corr:.4f}")

    # ✅ Plot Correlation Results
    plt.figure(figsize=(8, 5))
    bars = plt.bar(correlations.keys(), correlations.values(), color=["blue", "red", "green", "purple", "gray"])
    plt.xlabel("Learner Type")
    plt.ylabel("Correlation")
    plt.title("Correlation of Learners with Tesla Stock Data")
    plt.ylim(0, 1)

    # Add correlation labels
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

    plt.show()

    # ✅ Predict Future Tesla Stock Price
    predict_future_stock_price(learners["BagLearner (DT)"], X_raw)

# ✅ Run Script
if __name__ == "__main__":
    main()
