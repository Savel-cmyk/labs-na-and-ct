import mlflow
import mlflow.sklearn
import pandas as pd
import json

class WineClassifier:
    def __init__(self, model_uri="models:/WineClassifier/Production"):
        self.model_uri = model_uri
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        try:
            print(f"Загружаем sklearn модель из: {self.model_uri}")
            
            client = mlflow.MlflowClient()
            model_versions = client.search_model_versions(f"name='WineClassifier'")
            
            if model_versions:
                latest_version = max(model_versions, key=lambda x: x.version)
                model_uri = f"models:/WineClassifier/{latest_version.version}"
                print(f"Используем версию: {latest_version.version}")
            else:
                model_uri = self.model_uri
            
            self.pipeline = mlflow.sklearn.load_model(model_uri)
            
            print("Sklearn модель загружена")
            print(f"Тип модели: {type(self.pipeline)}")
            print(f"Классы: {self.pipeline.classes_ if hasattr(self.pipeline, 'classes_') else 'N/A'}")
            
        except Exception as e:
            print(f"Ошибка загрузки из MLflow: {e}")
            print("Пробуем загрузить локально...")
            self.load_local()
    
    def load_local(self):
        try:
            import joblib
            self.pipeline = joblib.load("wine_classifier_pipeline.joblib")
            print("Локальная модель загружена")
        except Exception as e:
            print(f"Ошибка локальной загрузки: {e}")
            print("Сначала обучите модель: python train_wine_model.py")
            raise
    
    def predict(self, wine_data):
        if isinstance(wine_data, dict):
            text = wine_data.get("description", "")
        else:
            text = str(wine_data)
        
        if not text.strip():
            raise ValueError("Текст описания вина не может быть пустым")
        
        print(f"Входной текст: {text[:50]}...")
        
        try:
            prediction = self.pipeline.predict([text])[0]
            
            if hasattr(self.pipeline, 'predict_proba'):
                probabilities = self.pipeline.predict_proba([text])[0]
                confidence = float(max(probabilities))
                probs_dict = dict(zip(self.pipeline.classes_, [float(p) for p in probabilities]))
            else:
                confidence = 1.0
                probs_dict = {prediction: 1.0}
            
            result = {
                "predicted_type": prediction,
                "confidence": confidence,
                "probabilities": probs_dict,
                "input_text": text[:100] + ("..." if len(text) > 100 else "")
            }
            
            print(f"Результат: {prediction} (уверенность: {confidence:.2%})")
            return result
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            print(f"Отладка - тип pipeline: {type(self.pipeline)}")
            print(f"Отладка - методы: {dir(self.pipeline)[:10]}...")
            raise

def test_model():
    mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
    
    print("Инициализируем классификатор...")
    
    try:
        classifier = WineClassifier()
        
        test_cases = [
            {"description": "Красное вино с нотами черной смородины"},
            {"description": "Белое вино с цитрусовыми нотами"},
            {"description": "Игристое вино с пузырьками"},
            {"description": "Розовое вино фруктовое"}
        ]
        
        print("\nЗапускаем тесты:")
        for i, test_data in enumerate(test_cases, 1):
            print(f"\nТест #{i}:")
            try:
                result = classifier.predict(test_data)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Ошибка: {e}")
                
    except Exception as e:
        print(f"Не удалось инициализировать классификатор: {e}")
        print("\nПопробуйте сначала обучить модель:")
        print("1. python train_wine_model.py")
        print("2. Проверьте что модель в MLflow: https://mlflow.labs.itmo.loc")

if __name__ == "__main__":
    test_model()