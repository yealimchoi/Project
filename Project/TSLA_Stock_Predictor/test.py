import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from DTLearner import DTLearner
from RTLearner import RTLearner
from BagLearner import BagLearner
from InsaneLearner import InsaneLearner

# Load dataset
data = np.loadtxt("Data/TSLA.csv", delimiter=",", skiprows=1, usecols=range(1, 7))  # Skip Date column
X = data[:, :-1]  # Features (Open, High, Low, Close, Volume)
Y = data[:, -1]   # Target (Adj Close)

# Train-test split (60% train, 40% test)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.6 * len(X))
train_X, test_X = X[indices[:train_size]], X[indices[train_size:]]
train_Y, test_Y = Y[indices[:train_size]], Y[indices[train_size:]]

# Function to evaluate models
def evaluate_model(learner, name):
    pred_Y = learner.query(test_X)
    corr = np.corrcoef(pred_Y, test_Y)[0, 1]
    mae = mean_absolute_error(test_Y, pred_Y)
    rmse = np.sqrt(mean_squared_error(test_Y, pred_Y))
    r2 = r2_score(test_Y, pred_Y)
    print(f"\n{name} Performance:")
    print(f"📌 Correlation: {corr:.4f}")
    print(f"📌 MAE: {mae:.4f}")
    print(f"📌 RMSE: {rmse:.4f}")
    print(f"📌 R² Score: {r2:.4f}")
    return corr, mae, rmse, r2

# Tune hyperparameters for DTLearner
best_leaf_size = None
best_corr = -1
for leaf_size in [5, 10, 20, 50, 100]:
    learner = DTLearner(leaf_size=leaf_size, verbose=False)
    learner.add_evidence(train_X, train_Y)
    corr, _, _, _ = evaluate_model(learner, f"DTLearner (Leaf={leaf_size})")
    if corr > best_corr:
        best_corr = corr
        best_leaf_size = leaf_size

print(f"\n✅ Best Leaf Size for DTLearner: {best_leaf_size} with correlation {best_corr:.4f}")

# Train models with optimal leaf_size
dt_learner = DTLearner(leaf_size=best_leaf_size, verbose=False)
dt_learner.add_evidence(train_X, train_Y)
evaluate_model(dt_learner, "DTLearner")

rt_learner = RTLearner(leaf_size=best_leaf_size, verbose=False)
rt_learner.add_evidence(train_X, train_Y)
evaluate_model(rt_learner, "RTLearner")

bag_dt = BagLearner(learner=DTLearner, kwargs={"leaf_size": best_leaf_size}, bags=20, verbose=False)
bag_dt.add_evidence(train_X, train_Y)
evaluate_model(bag_dt, "BagLearner (DT)")

bag_rt = BagLearner(learner=RTLearner, kwargs={"leaf_size": best_leaf_size}, bags=20, verbose=False)
bag_rt.add_evidence(train_X, train_Y)
evaluate_model(bag_rt, "BagLearner (RT)")

insane_learner = InsaneLearner(verbose=False)
insane_learner.add_evidence(train_X, train_Y)
evaluate_model(insane_learner, "InsaneLearner")

# Plot Predictions for Comparison
plt.figure(figsize=(12, 6))
plt.plot(test_Y, label="Actual", linestyle="dashed", alpha=0.7)
plt.plot(dt_learner.query(test_X), label="DTLearner", alpha=0.7)
plt.plot(bag_dt.query(test_X), label="BagLearner (DT)", alpha=0.7)
plt.plot(insane_learner.query(test_X), label="InsaneLearner", alpha=0.7)
plt.xlabel("Test Data Points")
plt.ylabel("Stock Price (Adj Close)")
plt.title("Comparison of Learner Predictions")
plt.legend()
plt.show()
