from src.prediction.predictor import build_prediction_frame


def test_prediction_frame_contains_engineered_features():
    frame = build_prediction_frame(
        {
            "temperature": 29,
            "humidity": 82,
            "pressure": 1002,
            "wind_speed": 18,
            "cloud_cover": 76,
            "sunshine_hours": 3,
            "month": 7,
        }
    )

    assert frame.shape == (1, 10)
    assert "pressure_drop_signal" in frame.columns

