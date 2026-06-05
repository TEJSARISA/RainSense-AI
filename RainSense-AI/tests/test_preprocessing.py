from src.preprocessing.data_loader import generate_sample_weather_data
from src.preprocessing.preprocessor import prepare_dataset
from src.preprocessing.validation import validate_weather_data


def test_generated_data_validates_and_prepares(tmp_path):
    data_path = tmp_path / "weather.csv"
    df = generate_sample_weather_data(data_path, rows=120, seed=7)
    report = validate_weather_data(df)
    prepared = prepare_dataset(df, save=False)

    assert report["missing_columns"] == []
    assert len(prepared) > 80
    assert "temp_humidity_index" in prepared.columns
    assert prepared["rain"].isin([0, 1]).all()

