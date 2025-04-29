from rf_interface import load_model, predict

# Load Pipeline
model = load_model()

# Features
new_features = [
    "isAmazon",
    "isFBA",
    "avg_price_rank_60d",
    "avg_price_rank_14d",
    "avg_price_rank_30d",
    "avg_self_price_30d",
    "avg_self_price_14d",
    "price_rank",
    "avg_self_price_60d",
    "price_diff",
]

# Call predict()
label, probability = predict(model, new_features)

# Outputs
print("Predicted class (0/1):", label)
print("Probability of class 1:", probability)
if label == 1:
    print("Win Buy Box")
else:
    print("No Buy Box")
