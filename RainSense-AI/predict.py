from __future__ import annotations

import argparse
import json

from src.prediction.predictor import predict_rainfall


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict rainfall from weather observations.")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--wind-speed", type=float, required=True)
    parser.add_argument("--cloud-cover", type=float, required=True)
    parser.add_argument("--sunshine-hours", type=float, required=True)
    parser.add_argument("--month", type=int, default=7)
    args = parser.parse_args()
    payload = {
        "temperature": args.temperature,
        "humidity": args.humidity,
        "pressure": args.pressure,
        "wind_speed": args.wind_speed,
        "cloud_cover": args.cloud_cover,
        "sunshine_hours": args.sunshine_hours,
        "month": args.month,
    }
    print(json.dumps(predict_rainfall(payload), indent=2))


if __name__ == "__main__":
    main()

