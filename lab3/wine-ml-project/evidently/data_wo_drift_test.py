import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite

from evidently.tests import (
    TestShareOfDriftedColumns,
    TestNumberOfDriftedColumns,
    TestColumnDrift,
    TestColumnShareOfMissingValues,
    TestColumnQuantile
)
import json
import os

class WineDataDriftMonitor:
    """
    Класс для мониторинга дрейфа данных в модели классификации вин.
    Совместим с evidently 0.6.7
    """
    
    def __init__(self, reference_data_path=None):
        """
        Инициализация мониторинга дрейфа.
        
        Args:
            reference_data_path: Путь к файлу с эталонными данными
        """
        mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
        mlflow.set_experiment("wine-drift-monitoring")
        
        self.column_mapping = ColumnMapping()
        self.column_mapping.text_features = ['description']
        self.column_mapping.target = 'type'
        self.column_mapping.prediction = 'prediction'
        self.reference_data = self.load_or_create_reference_data(reference_data_path)
        
        print(f"Монитор дрейфа инициализирован (evidently 0.6.7)")
        print(f"Эталонных записей: {len(self.reference_data)}")
    
    def load_or_create_reference_data(self, reference_data_path):
        """
        Загружает эталонные данные из файла или создает их.
        """
        if reference_data_path and os.path.exists(reference_data_path):
            print(f"Загрузка эталонных данных из {reference_data_path}")
            return pd.read_csv(reference_data_path)
        else:
            print("Создание эталонных данных...")
            return self.create_reference_dataset()
    
    def create_reference_dataset(self, num_samples=500):
        """
        Создает эталонный датасет для сравнения.
        """
        wine_types = ["Красное", "Белое", "Игристое", "Розовое"]
        
        red_descriptions = [
            "красное вино с нотами черной смородины и вишни",
            "элегантное красное вино с танинами",
            "насыщенное красное вино с ягодными нотами",
            "сухое красное вино с ароматом дуба",
            "красное вино с тонами шоколада и специй"
        ]
        
        white_descriptions = [
            "белое вино с цитрусовыми нотами",
            "свежее белое вино с минеральными тонами",
            "сладкое белое вино с ароматом меда",
            "белое вино с нотами зеленого яблока",
            "богатое белое вино с ванильными оттенками"
        ]
        
        sparkling_descriptions = [
            "игристое вино с мелкими пузырьками",
            "шампанское брют с цветочным ароматом",
            "игристое вино с тонами дрожжей",
            "праздничное игристое вино",
            "игристое вино с яблочными нотами"
        ]
        
        rose_descriptions = [
            "розовое вино с ягодными нотами",
            "свежее розовое вино фруктовое",
            "розовое полусладкое вино",
            "розовое вино с оттенками клубники",
            "элегантное розовое вино"
        ]
        
        data = []
        start_date = datetime.now() - timedelta(days=90)
        
        for i in range(num_samples):
            wine_type = random.choice(wine_types)
            
            if wine_type == "Красное":
                description = random.choice(red_descriptions)
            elif wine_type == "Белое":
                description = random.choice(white_descriptions)
            elif wine_type == "Игристое":
                description = random.choice(sparkling_descriptions)
            else:
                description = random.choice(rose_descriptions)
            
            if random.random() > 0.7:
                adjectives = ["выдержанное", "молодое", "органическое", "традиционное", "современное"]
                description = f"{random.choice(adjectives)} {description}"
            
            timestamp = start_date + timedelta(days=random.randint(0, 90), 
                                             hours=random.randint(0, 23))
            
            text_length = len(description)
            word_count = len(description.split())
            
            data.append({
                'description': description,
                'type': wine_type,
                'timestamp': timestamp,
                'text_length': text_length,
                'word_count': word_count,
                'avg_word_length': text_length / word_count if word_count > 0 else 0,
                'contains_fruit': int('ягод' in description or 'фрукт' in description or 'яблок' in description),
                'contains_spice': int('специ' in description or 'перец' in description or 'кориц' in description),
                'is_dry': int('сух' in description.lower() or 'брют' in description.lower())
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['prediction'] = df['type']
        
        df.to_csv('reference_wine_data.csv', index=False)
        print(f"Эталонные данные сохранены в reference_wine_data.csv")
        
        return df
    
    def generate_current_data(self, num_samples=200, drift_intensity=0.0):
        """
        Генерирует текущие данные для сравнения.
        
        Args:
            num_samples: Количество образцов
            drift_intensity: Интенсивность дрейфа (0.0 - нет дрейфа, 1.0 - сильный дрейф)
        """
        print(f"\nГенерация текущих данных (дрейф: {drift_intensity*100:.1f}%)...")
        
        wine_types = ["Красное", "Белое", "Игристое", "Розовое"]
        
        new_descriptions = {
            "Красное": [
                "натуральное красное вино биодинамическое",
                "красное вино с нотами трюфеля и кожи",
                "крафтовое красное вино с дикими дрожжами",
                "красное вино с тонами кофе и темного шоколада"
            ],
            "Белое": [
                "апельсиновое вино с кожурой",
                "белое вино с нотами устриц и моря",
                "натуральное белое вино без фильтрации",
                "белое вино с тонами дыни и тропических фруктов"
            ],
            "Игристое": [
                "петнат натуральный игристый",
                "игристое вино с длительной выдержкой на осадке",
                "игристое вино методом шарма",
                "игристое вино с нотами бриоши и орехов"
            ],
            "Розовое": [
                "розовое вино с нотами розы и лепестков",
                "розовое вино прованского стиля",
                "розовое вино с тонами граната и клюквы",
                "розовое вино с минеральным характером"
            ]
        }
        
        data = []
        start_date = datetime.now() - timedelta(days=7)
        
        for i in range(num_samples):
            if random.random() < drift_intensity:
                wine_type = random.choice(["Игристое", "Розовое"])
            else:
                wine_type = random.choice(wine_types)
            
            if random.random() < drift_intensity * 0.7:
                description = random.choice(new_descriptions[wine_type])
            else:
                ref_samples = self.reference_data[self.reference_data['type'] == wine_type]
                if len(ref_samples) > 0:
                    description = ref_samples.sample(1)['description'].iloc[0]
                else:
                    description = f"{wine_type.lower()} вино"
            
            if random.random() < drift_intensity * 0.5:
                description = description + " " + " ".join(["очень"] * random.randint(1, 3))
            
            timestamp = start_date + timedelta(days=random.randint(0, 7), 
                                             hours=random.randint(0, 23))
            
            text_length = len(description)
            word_count = len(description.split())
            
            data.append({
                'description': description,
                'type': wine_type if random.random() > 0.1 else random.choice(wine_types),
                'timestamp': timestamp,
                'text_length': text_length,
                'word_count': word_count,
                'avg_word_length': text_length / word_count if word_count > 0 else 0,
                'contains_fruit': int('ягод' in description or 'фрукт' in description or 
                                     'яблок' in description or 'цитрус' in description),
                'contains_spice': int('специ' in description or 'перец' in description or 
                                     'кориц' in description or 'кофе' in description),
                'is_dry': int('сух' in description.lower() or 'брют' in description.lower())
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['prediction'] = df.apply(
            lambda row: row['type'] if random.random() > drift_intensity * 0.3 
            else random.choice([t for t in wine_types if t != row['type']]), 
            axis=1
        )
        
        return df
    
    def run_drift_analysis(self, current_data, run_name=None):
        """
        Запускает полный анализ дрейфа данных с Evidently 0.6.7.
        
        Args:
            current_data: Текущие данные для анализа
            run_name: Имя запуска в MLflow
        
        Returns:
            dict: Результаты анализа
        """
        if run_name is None:
            run_name = f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"\n{'='*60}")
            print(f"Запуск анализа дрейфа: {run_name}")
            print(f"{'='*60}")
            
            mlflow.log_param("analysis_date", datetime.now().isoformat())
            mlflow.log_param("reference_samples", len(self.reference_data))
            mlflow.log_param("current_samples", len(current_data))
            
            print("\n1. Анализ качества данных...")
            data_quality_report = Report(metrics=[DataQualityPreset()])
            data_quality_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            quality_report_path = "data_quality_report.html"
            data_quality_report.save_html(quality_report_path)
            mlflow.log_artifact(quality_report_path)
            print(f"Отчет качества сохранен: {quality_report_path}")
            
            print("\n2. Анализ дрейфа данных...")
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            drift_report_path = "data_drift_report.html"
            data_drift_report.save_html(drift_report_path)
            mlflow.log_artifact(drift_report_path)
            print(f"Отчет дрейфа сохранен: {drift_report_path}")
            
            print("\n3. Запуск автоматических тестов на дрейф...")
            drift_test_suite = TestSuite(tests=[
				TestShareOfDriftedColumns(lt=0.3), 
				
				TestNumberOfDriftedColumns(lt=5),   
				
				TestColumnDrift(column_name='text_length'),
				TestColumnDrift(column_name='word_count'),
				TestColumnDrift(column_name='contains_fruit'),
				TestColumnDrift(column_name='contains_spice'),
				
				TestColumnShareOfMissingValues(column_name='description', lt=0.05),
			])
            
            drift_test_suite.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            test_results = drift_test_suite.as_dict()
            test_results_path = "drift_test_results.json"
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
            mlflow.log_artifact(test_results_path)
            
            all_tests_passed = True
            failed_tests = []
            
            for test in test_results['tests']:
                test_name = test['name']
                test_status = test['status']
                test_group = test['group']
                
                if 'value' in test.get('result', {}):
                    mlflow.log_metric(f"test_{test_name}", test['result']['value'])
                
                if test_status == 'FAIL':
                    all_tests_passed = False
                    failed_tests.append({
                        'name': test_name,
                        'group': test_group,
                        'details': test.get('result', {})
                    })
                    print(f"Тест не пройден: {test_name} ({test_group})")
                else:
                    print(f"Тест пройден: {test_name} ({test_group})")
            
            print("\n4. Анализ метрик дрейфа...")
            drift_metrics = data_drift_report.as_dict()
            
            if 'metrics' in drift_metrics:
                for metric in drift_metrics['metrics']:
                    if metric['metric'] == 'DatasetDriftMetric':
                        drift_score = metric['result'].get('drift_score', 0)
                        drift_detected = metric['result'].get('drift_detected', False)
                        n_drifted_features = metric['result'].get('number_of_drifted_features', 0)
                        
                        mlflow.log_metric("dataset_drift_score", drift_score)
                        mlflow.log_metric("n_drifted_features", n_drifted_features)
                        mlflow.log_param("dataset_drift_detected", drift_detected)
                        
                        if drift_detected:
                            print(f"Обнаружен дрейф в наборе данных!")
                            print(f"Score: {drift_score:.3f}, Дрейфующих признаков: {n_drifted_features}")
            
            mlflow.log_param("all_tests_passed", all_tests_passed)
            mlflow.log_param("failed_tests_count", len(failed_tests))
            mlflow.log_param("drift_analysis_complete", True)
            
            current_data_path = "current_wine_data.csv"
            current_data.to_csv(current_data_path, index=False)
            mlflow.log_artifact(current_data_path)
            
            print(f"\n{'='*60}")
            if all_tests_passed:
                print("Все тесты пройдены. Значительный дрейф не обнаружен.")
                mlflow.log_param("drift_conclusion", "No significant drift detected")
            else:
                print(f"Обнаружен дрейф! Не пройдено тестов: {len(failed_tests)}")
                mlflow.log_param("drift_conclusion", f"Drift detected in {len(failed_tests)} tests")
                
                print("\nДетали проваленных тестов:")
                for test in failed_tests:
                    print(f"  - {test['name']}: {test.get('details', {})}")
            
            print(f"{'='*60}")
            
            analysis_result = {
                'run_name': run_name,
                'timestamp': datetime.now().isoformat(),
                'all_tests_passed': all_tests_passed,
                'failed_tests': failed_tests,
                'reference_samples': len(self.reference_data),
                'current_samples': len(current_data),
                'reports': {
                    'quality_report': quality_report_path,
                    'drift_report': drift_report_path,
                    'test_results': test_results_path
                }
            }
            
            return analysis_result

def main():
    """
    Основная функция для запуска анализа дрейфа.
    """
    print("Запуск системы мониторинга дрейфа данных для классификации вин")
    print("="*70)
    
    try:
        monitor = WineDataDriftMonitor()
        
        print("\nГенерация текущих данных (без искусственного дрейфа)...")
        current_data = monitor.generate_current_data(num_samples=200, drift_intensity=0.0)
        
        analysis_result = monitor.run_drift_analysis(current_data)
        
        print("\n" + "="*70)
        print("Анализ дрейфа завершен успешно!")
        
    except Exception as e:
        print(f"\nОшибка при выполнении анализа дрейфа: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())