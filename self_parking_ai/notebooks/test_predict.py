from utils.strategy_predictor import predict_strategy

# Example test case
strategy, confidence, scores = predict_strategy(3.0, 1.6, 5.0, 2.5, 1.0, verbose=True)