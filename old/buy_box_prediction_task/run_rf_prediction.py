from rf_interface import load_pipeline, predict

# Load Pipeline
model = load_pipeline()

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

test_features = [
    0,  # isAmazon
    1,  # isFBA
    0.5,  # avg_price_rank_60d: Mean of this seller’s price_rank over the past 60 days
    0.6,  # avg_price_rank_14d
    0.7,  # avg_price_rank_30d
    2.5,  # avg_self_price_30d: Mean of this seller’s price over the past 30 days.
    3.0,  # avg_self_price_14d
    2.0,  # price_rank: Rank of this seller’s price at time t. Higer rank means lower price.
    2.0,  # avg_self_price_60d
    -1.5,  # price_diff: Difference between this seller’s price and the minimum price at time.
]
# Call predict()
label, probability = predict(model, test_features)

# Outputs
print("Predicted class (0/1):", label)
print("Probability of class 1:", probability)
if label == 1:
    print("Win Buy Box")
else:
    print("No Buy Box")
