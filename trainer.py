 # trainer.py

import os
from learner import train_model, generate_strategies, save_model
from strategies import load_strategies, add_generated_strategies

MODEL_PATH = "models/strategy_model.pkl"

def run_training_pipeline():
    strategies_data = load_strategies()
    user_strategies = strategies_data["user"]

    if len(user_strategies) < 5:
        raise ValueError("Need 5 user strategies before training.")

    print("[*] Training model on user strategies...")
    model_data = train_model(user_strategies, MODEL_PATH)

    print("[*] Generating new strategies using AI model...")
    generated = generate_strategies(model_data, user_strategies, num=3)

    print("[*] Saving generated strategies...")
    add_generated_strategies(generated)

    print("[✓] Training complete. 3 new AI strategies created.")
    return generated