import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import random
from typing import List, Dict
import uuid

def generate_wine_description(style: str) -> str:
    """Генерирует описание вина для заданного стиля"""
    
    wine_descriptions = {
        "Красное": [
            "красное вино с нотами {} и {}, с {}",
            "элегантное красное вино, {} с {}",
            "интенсивное красное вино, {} с оттенками {}",
            "бархатистое красное вино с ароматами {} и {}"
        ],
        "Белое": [
            "белое вино с ароматами {} и {}, {}",
            "свежее белое вино, {} с {}",
            "богатое белое вино с тонами {} и {}",
            "минеральное белое вино, {}"
        ],
        "Игристое": [
            "игристое вино с {} и {}, {}",
            "шампанское с нотами {} и {}",
            "игристый напиток, {} с {}",
            "праздничное игристое вино с ароматами {}"
        ],
        "Розовое": [
            "розовое вино с {} и {}, {}",
            "свежее розовое вино, {} с {}",
            "фруктовое розовое вино с оттенками {}",
            "элегантное розовое вино с ароматами {} и {}"
        ]
    }
    
    red_notes = ["черной смородины", "вишни", "ежевики", "сливы", "ванили", "дуба", "табака", 
                "шоколада", "переца", "корицы", "кожи", "землистых тонов"]
    
    white_notes = ["цитрусовых", "зеленого яблока", "груши", "персика", "абрикоса", "меда",
                  "минеральных нот", "цветочных тонов", "тропических фруктов", "ванили"]
    
    sparkling_notes = ["мелких пузырьков", "дрожжей", "хлебных корочек", "яблока", "груши",
                      "цветочных ароматов", "цитрусовых", "миндаля", "меда"]
    
    rose_notes = ["клубники", "малины", "ежевики", "арбуза", "цитрусовых", "цветов", 
                 "зеленых нот", "специй", "ванили"]
    
    taste_adjectives = ["сухое", "полусухое", "полусладкое", "сладкое", "крепкое", 
                       "бархатистое", "танинное", "кислое", "сбалансированное"]
    
    structure_words = ["долгое послевкусие", "средней полноты", "полное тело", 
                      "нежное", "мощное", "элегантное", "комплексное"]
    
    template = random.choice(wine_descriptions[style])
    
    if style == "Красное":
        notes = random.sample(red_notes, 2)
    elif style == "Белое":
        notes = random.sample(white_notes, 2)
    elif style == "Игристое":
        notes = random.sample(sparkling_notes, 2)
    else:
        notes = random.sample(rose_notes, 2)
    
    description = template.format(notes[0], notes[1], 
                                 random.choice(structure_words))
    
    if random.random() > 0.5:
        description = f"{random.choice(taste_adjectives)} {description}"
    
    return description.capitalize()

def generate_realistic_training_data(num_samples: int = 200) -> pd.DataFrame:
    """Генерирует реалистичные тренировочные данные"""
    
    wine_types = ["Красное", "Белое", "Игристое", "Розовое"]
    
    data = []
    
    samples_per_class = num_samples // len(wine_types)
    
    for wine_type in wine_types:
        for _ in range(samples_per_class):
            for _ in range(3):
                description = generate_wine_description(wine_type)
                
                if random.random() > 0.7:
                    extra_words = ["выдержанное", "молодое", "органическое", 
                                 "биодинамическое", "традиционное"]
                    description = f"{random.choice(extra_words)} {description}"
                
                data.append({
                    "text": description,
                    "type": wine_type
                })
    
    borderline_cases = [
        {"text": "Золотистое вино с нотками дуба и ванили, насыщенное", "type": "Белое"},
        {"text": "Темное вино с фруктовыми нотами и танинами", "type": "Красное"},
        {"text": "Светлое игристое вино с фруктовыми нотами", "type": "Игристое"},
        {"text": "Бледно-розовое вино со свежими ягодными нотами", "type": "Розовое"},
        {"text": "Вино янтарного цвета с богатым ароматом", "type": "Белое"},
        {"text": "Рубиновое вино с пряными нотами", "type": "Красное"},
        {"text": "Игристое розовое вино с мелкими пузырьками", "type": "Игристое"},
        {"text": "Нежно-розовое вино с цветочными ароматами", "type": "Розовое"},
    ]
    
    data.extend(borderline_cases)
    
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    
    print(f"Сгенерировано {len(df)} уникальных примеров")
    print(f"Распределение по классам:\n{df['type'].value_counts()}")
    
    return df

def train_model():
    mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
    mlflow.set_experiment("wine-classification")
    
    with mlflow.start_run():
        df = generate_realistic_training_data(num_samples=200)
        
        X = df['text']
        y = df['type']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000, 
                stop_words=None,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced'
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_param("max_features", 1000)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")
        
        for class_name in pipeline.classes_:
            if class_name in report:
                mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
                mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])
        
        input_example = {
            "description": "Элегантное красное вино с нотами черной смородины и вишни"
        }
        
        mlflow.sklearn.log_model(
            pipeline,
            "wine_classifier",
            input_example=input_example,
            registered_model_name="WineClassifier"
        )
        
        joblib.dump(pipeline, "wine_classifier_pipeline.joblib")
        mlflow.log_artifact("wine_classifier_pipeline.joblib")
        
        test_samples = pd.DataFrame({
            'text': X_test[:10],
            'actual': y_test[:10],
            'predicted': y_pred[:10]
        })
        test_samples.to_csv("test_samples.csv", index=False)
        mlflow.log_artifact("test_samples.csv")
        
        print("\n" + "="*50)
        print("Модель обучена успешно!")
        print(f"Точность (accuracy): {accuracy:.4f}")
        print(f"Классы: {pipeline.classes_}")
        print(f"Количество признаков: {len(pipeline.named_steps['tfidf'].get_feature_names_out())}")
        print("="*50)
        
        test_cases = [
            {"description": "Элегантное красное вино с нотами черной смородины, вишни и легкими танинами"},
            {"description": "Свежее белое вино с цитрусовыми и минеральными нотами"},
            {"description": "Игристое вино с мелкими пузырьками и цветочным ароматом"},
            {"description": "Фруктовое розовое вино с ягодными нотами"},
            {"description": "Сложное для классификации описание вина"}
        ]
        
        print("\nТестирование на примерах формата приложения:")
        for test_case in test_cases:
            text = test_case["description"]
            prediction = pipeline.predict([text])[0]
            probability = pipeline.predict_proba([text])[0]
            
            print(f"\nВход: '{text[:60]}...'")
            print(f"Предсказание: {prediction}")
            print(f"Вероятности: {dict(zip(pipeline.classes_, [f'{p:.3f}' for p in probability]))}")
        
        print(f"\nМодель сохранена в MLflow Registry как 'WineClassifier'")
        print(f"URI для загрузки в Spring Boot: 'models:/WineClassifier/Production'")

if __name__ == "__main__":
    train_model()