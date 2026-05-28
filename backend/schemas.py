from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    time: float = Field(..., alias="time", description="Временная метка")
    android_sensor_accelerometer_mean: float = Field(..., alias="android.sensor.accelerometer#mean")
    android_sensor_accelerometer_min: float = Field(..., alias="android.sensor.accelerometer#min")
    android_sensor_accelerometer_max: float = Field(..., alias="android.sensor.accelerometer#max")
    android_sensor_accelerometer_std: float = Field(..., alias="android.sensor.accelerometer#std")
    android_sensor_game_rotation_vector_mean: float = Field(..., alias="android.sensor.game_rotation_vector#mean")
    android_sensor_game_rotation_vector_min: float = Field(..., alias="android.sensor.game_rotation_vector#min")
    android_sensor_game_rotation_vector_max: float = Field(..., alias="android.sensor.game_rotation_vector#max")
    android_sensor_game_rotation_vector_std: float = Field(..., alias="android.sensor.game_rotation_vector#std")
    android_sensor_gyroscope_mean: float = Field(..., alias="android.sensor.gyroscope#mean")
    android_sensor_gyroscope_min: float = Field(..., alias="android.sensor.gyroscope#min")
    android_sensor_gyroscope_max: float = Field(..., alias="android.sensor.gyroscope#max")
    android_sensor_gyroscope_std: float = Field(..., alias="android.sensor.gyroscope#std")
    android_sensor_gyroscope_uncalibrated_mean: float = Field(..., alias="android.sensor.gyroscope_uncalibrated#mean")
    android_sensor_gyroscope_uncalibrated_min: float = Field(..., alias="android.sensor.gyroscope_uncalibrated#min")
    android_sensor_gyroscope_uncalibrated_max: float = Field(..., alias="android.sensor.gyroscope_uncalibrated#max")
    android_sensor_gyroscope_uncalibrated_std: float = Field(..., alias="android.sensor.gyroscope_uncalibrated#std")
    android_sensor_linear_acceleration_mean: float = Field(..., alias="android.sensor.linear_acceleration#mean")
    android_sensor_linear_acceleration_min: float = Field(..., alias="android.sensor.linear_acceleration#min")
    android_sensor_linear_acceleration_max: float = Field(..., alias="android.sensor.linear_acceleration#max")
    android_sensor_linear_acceleration_std: float = Field(..., alias="android.sensor.linear_acceleration#std")
    android_sensor_orientation_mean: float = Field(..., alias="android.sensor.orientation#mean")
    android_sensor_orientation_min: float = Field(..., alias="android.sensor.orientation#min")
    android_sensor_orientation_max: float = Field(..., alias="android.sensor.orientation#max")
    android_sensor_orientation_std: float = Field(..., alias="android.sensor.orientation#std")
    android_sensor_rotation_vector_mean: float = Field(..., alias="android.sensor.rotation_vector#mean")
    android_sensor_rotation_vector_min: float = Field(..., alias="android.sensor.rotation_vector#min")
    android_sensor_rotation_vector_max: float = Field(..., alias="android.sensor.rotation_vector#max")
    android_sensor_rotation_vector_std: float = Field(..., alias="android.sensor.rotation_vector#std")
    sound_mean: float = Field(..., alias="sound#mean")
    sound_min: float = Field(..., alias="sound#min")
    sound_max: float = Field(..., alias="sound#max")
    sound_std: float = Field(..., alias="sound#std")
    speed_mean: float = Field(..., alias="speed#mean")
    speed_min: float = Field(..., alias="speed#min")
    speed_max: float = Field(..., alias="speed#max")
    speed_std: float = Field(..., alias="speed#std")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "time": 78.0,
                "android.sensor.accelerometer#mean": 9.81147629156,
                "android.sensor.accelerometer#min": 9.75889547269,
                "android.sensor.accelerometer#max": 9.84941095639,
                "android.sensor.accelerometer#std": 0.0146255073331,
                "android.sensor.game_rotation_vector#mean": 0.0293404701347,
                "android.sensor.game_rotation_vector#min": 0.0290141820322,
                "android.sensor.game_rotation_vector#max": 0.0295264272136,
                "android.sensor.game_rotation_vector#std": 0.00011870091454,
                "android.sensor.gyroscope#mean": 0.00165077522136,
                "android.sensor.gyroscope#min": 0.0,
                "android.sensor.gyroscope#max": 0.00353308255853,
                "android.sensor.gyroscope#std": 0.000736963641413,
                "android.sensor.gyroscope_uncalibrated#mean": 0.0162213599906,
                "android.sensor.gyroscope_uncalibrated#min": 0.0141724221314,
                "android.sensor.gyroscope_uncalibrated#max": 0.0186953157594,
                "android.sensor.gyroscope_uncalibrated#std": 0.000981864545925,
                "android.sensor.linear_acceleration#mean": 0.0209778383629,
                "android.sensor.linear_acceleration#min": 0.00249536633909,
                "android.sensor.linear_acceleration#max": 0.0524095823954,
                "android.sensor.linear_acceleration#std": 0.0110446872158,
                "android.sensor.orientation#mean": 354.286933099,
                "android.sensor.orientation#min": 353.598334711,
                "android.sensor.orientation#max": 354.942707018,
                "android.sensor.orientation#std": 0.2456763287,
                "android.sensor.rotation_vector#mean": 0.050413140269,
                "android.sensor.rotation_vector#min": 0.0447771592556,
                "android.sensor.rotation_vector#max": 0.0563513538347,
                "android.sensor.rotation_vector#std": 0.00210922613983,
                "sound#mean": 0.0,
                "sound#min": 0.0,
                "sound#max": 0.0,
                "sound#std": 0.0,
                "speed#mean": 0.0,
                "speed#min": 0.0,
                "speed#max": 0.0,
                "speed#std": 0.0
            }
        }
    }

class PredictionResponse(BaseModel):
    transport_mode: str = Field(..., description="Распознанный режим движения")
    probabilities: dict[str, float] = Field(..., description="Вероятности для каждого из 5 классов")
    status: str = Field(..., description="Статус инференса")
