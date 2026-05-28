import os
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
import dotenv
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Human Activity Recognition Classifier inference"
)

IN_FEATURES = 34
CLASSES_AMOUNT = 5
CLASS_NAMES = ['Bus', 'Car', 'Still', 'Train', 'Walking']

dotenv.load_dotenv(dotenv.find_dotenv())

# путь к сериализованному файлу предобработчика
PREPROCESSOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "preprocessor.pkl"
)

def preprocess_input(request_dict: dict, percentiles: dict, scaler: StandardScaler, feature_columns: list):
    df = pd.DataFrame([request_dict])
    
    cols_to_drop = [
        "time", 
        "android.sensor.orientation#min", 
        "android.sensor.orientation#max", 
        "android.sensor.orientation#mean", 
        "android.sensor.orientation#std"
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors="ignore")
    
    df["is_gps_active"] = (df["speed#mean"] > 0).astype(int)
    df["is_mic_active"] = (df["sound#mean"] > 0).astype(int)
    
    skewed_features_raw = (
        "android.sensor.gyroscope", 
        "speed", 
        "android.sensor.gyroscope_uncalibrated", 
        "android.sensor.linear_acceleration"
    )
    skewed_features = [f"{x}#{j}" for x in skewed_features_raw for j in ("mean", "min", "max", "std")]
    for feature in skewed_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature])
            
    for feature, (p1, p99) in percentiles.items():
        if feature in df.columns:
            df[feature] = np.where(df[feature] > p1, df[feature], p1)
            df[feature] = np.where(df[feature] < p99, df[feature], p99)
            
    df = df[feature_columns]
    scaled_values = scaler.transform(df)
    return scaled_values

# инициализация предобработчика из сериализованного файла
try:
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Файл предобработчика preprocessor.pkl не найден по пути: {PREPROCESSOR_PATH}")
    with open(PREPROCESSOR_PATH, "rb") as f:
        PERCENTILES, SCALER, FEATURE_COLUMNS = pickle.load(f)
    print("Успешно загружен сериализованный предобработчик признаков preprocessor.pkl.")
except Exception as e:
    print(f"Ошибка при загрузке предобработчика признаков: {e}")
    PERCENTILES, SCALER, FEATURE_COLUMNS = {}, None, []

# подключение к MLFlow, загрузка модели из регистра
model = None
model_source = "not_loaded"

try:
    dh_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    dh_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    dh_project_name = os.environ.get("MLFLOW_TRACKING_PROJECTNAME")

    mlflow.set_tracking_uri(f"https://dagshub.com/{dh_username}/{dh_project_name}.mlflow")
    
    model_name = "MLP_SensorClassifier"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    
    print(f"Попытка загрузить зарегистрированную модель '{model_name}' из MLflow...")
    model = mlflow.pyfunc.load_model(model_uri)
    model_source = "mlflow"
    print(f"Успешно загружена модель '{model_name}' (версия: {model_version}) из MLflow Model Registry.")
except Exception as e:
    print(f"Не удалось загрузить модель из MLflow ({e}). Переключение на локальный чекпоинт...")

# загрузка модели из локального чекпоинта
if model is None:
    from backend.model import MLPClassifier
    
    local_model = MLPClassifier(in_dim=IN_FEATURES, hidden_dims=(112, 112), out_dim=CLASSES_AMOUNT)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "checkpoints")
    
    ckpt = tf.train.Checkpoint(model=local_model)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    
    if latest_ckpt:
        ckpt.restore(latest_ckpt).expect_partial()
        model_source = "local_checkpoint"
        print(f"Успешно загружены веса модели из локального чекпоинта: {latest_ckpt}")
        
        # приведение интерфейса predict к виду, идентичному PyFuncModel
        class LocalModelWrapper:
            def __init__(self, tf_model):
                self.tf_model = tf_model
            def predict(self, features):
                input_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
                return self.tf_model.predict(input_tensor).numpy()
                
        model = LocalModelWrapper(local_model)
    else:
        print(f"Локальный чекпоинт не найден в '{checkpoint_dir}'. Инференс будет на случайных весах.")
        model_source = "random_weights"
        
        class LocalModelWrapper:
            def __init__(self, tf_model):
                self.tf_model = tf_model
            def predict(self, features):
                input_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
                return self.tf_model.predict(input_tensor).numpy()
                
        model = LocalModelWrapper(local_model)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Модель не инициализирована")
        if SCALER is None:
            raise HTTPException(status_code=503, detail="Предобработчик признаков не инициализирован")
            
        # преобразование Pydantic модель в словарь с алиасами
        request_dict = request.model_dump(by_alias=True) if hasattr(request, "model_dump") else request.dict(by_alias=True)
        
        # предобработка признаков
        scaled_features = preprocess_input(request_dict, PERCENTILES, SCALER, FEATURE_COLUMNS)
        
        # MLPWrapper и LocalModelWrapper ожидают на вход двумерный массив
        probabilities = model.predict(scaled_features)[0]
        
        prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(CLASSES_AMOUNT)}
        
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        if model_source == "mlflow":
            status_msg = "prediction successful (loaded from mlflow)"
        elif model_source == "local_checkpoint":
            status_msg = "prediction successful (loaded from local checkpoint)"
        else:
            status_msg = "prediction warning: random weights (model/checkpoint not found)"
            
        return PredictionResponse(
            transport_mode=predicted_class,
            probabilities=prob_dict,
            status=status_msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при инференсе: {str(e)}")
