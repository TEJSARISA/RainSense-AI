from src.training.train_models import train_all_models


if __name__ == "__main__":
    result = train_all_models()
    print(f"Best model: {result['best_model']}")

