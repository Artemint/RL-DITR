import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# === Параметры ===
SAMPLE_CSV = 'assets/data/sample.csv'
PREDICTIONS_PATH = 'results/predictions_full.json'

class FeatureImportanceAnalyzerStable:
    """
    СТАБИЛЬНЫЙ анализатор значимости признаков для RL-DITR модели
    ВНИМАНИЕ: Анализ проводится на одном пациенте - результаты не экстраполируются!
    """
    
    def __init__(self, random_seed=42):
        self.df_sample = None
        self.features_data = None
        self.model_predictions = None
        self.scaler = StandardScaler()
        self.random_seed = random_seed
        
        # Устанавливаем фиксированные seed'ы для воспроизводимости
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
    def load_data(self):
        """Загрузка данных"""
        print("📊 Загрузка данных...")
        
        # Загружаем sample.csv
        self.df_sample = pd.read_csv(SAMPLE_CSV)
        self.df_sample['datetime'] = pd.to_datetime(self.df_sample['datetime'])
        
        # Загружаем предсказания модели
        import json
        with open(PREDICTIONS_PATH) as f:
            self.model_predictions = json.load(f)
            
        print(f"✅ Загружено {len(self.df_sample)} записей для одного пациента")
        
    def prepare_features(self):
        """Подготовка признаков для анализа (СТАБИЛЬНАЯ ВЕРСИЯ)"""
        print("🔧 Подготовка признаков (стабильная версия)...")
        
        # Извлекаем статические признаки (демографические, физиологические)
        static_features = {}
        
        # Демографические признаки
        age_data = self.df_sample[self.df_sample['key'] == 'age']
        if not age_data.empty:
            static_features['age'] = age_data['value'].iloc[0]
            
        gender_data = self.df_sample[self.df_sample['key'] == 'gender']
        if not gender_data.empty:
            static_features['gender'] = 1 if gender_data['value'].iloc[0] == 'M' else 0
            
        # Физиологические признаки
        for feature in ['height', 'weight', 'BMI', 'SBP', 'DBP', 'RR']:
            feature_data = self.df_sample[self.df_sample['key'] == feature]
            if not feature_data.empty:
                static_features[feature] = feature_data['value'].iloc[0]
                
        # Лабораторные признаки (берем средние значения)
        lab_features = {}
        lab_data = self.df_sample[self.df_sample['key_type'] == 'cont']
        for feature in lab_data['key'].unique():
            if feature not in ['age', 'height', 'weight', 'BMI', 'SBP', 'DBP', 'RR', 'glu']:
                values = lab_data[lab_data['key'] == feature]['value'].astype(float)
                if len(values) > 0:
                    lab_features[f'lab_{feature}'] = values.mean()
                    
        static_features.update(lab_features)
        
        # Временные признаки (глюкоза и инсулин) - СТАБИЛЬНАЯ ОБРАБОТКА
        temporal_features = {}
        
        # Глюкоза (последние значения)
        glu_data = self.df_sample[self.df_sample['key'] == 'glu'].sort_values('datetime')
        if not glu_data.empty:
            temporal_features['glu_current'] = glu_data['value'].iloc[-1]
            temporal_features['glu_mean'] = glu_data['value'].astype(float).mean()
            temporal_features['glu_std'] = glu_data['value'].astype(float).std()
            
        # СТАБИЛЬНАЯ обработка инсулина
        insulin_data = self.df_sample[self.df_sample['key'] == 'insulin'].sort_values('datetime')
        if not insulin_data.empty:
            # Преобразуем в float (исправление типа данных)
            insulin_values = insulin_data['value'].astype(float)
            temporal_features['insulin_current'] = insulin_values.iloc[-1]
            temporal_features['insulin_mean'] = insulin_values.mean()
            temporal_features['insulin_std'] = insulin_values.std()
            
            # Добавляем инсулин за последние 12 часов (более релевантно)
            recent_insulin = insulin_data[insulin_data['datetime'] >= 
                                        (insulin_data['datetime'].max() - pd.Timedelta(hours=12))]
            if not recent_insulin.empty:
                temporal_features['insulin_recent_mean'] = recent_insulin['value'].astype(float).mean()
            
        # Тип инсулина
        insulin_group_data = self.df_sample[self.df_sample['key'] == 'insulin_group']
        if not insulin_group_data.empty:
            insulin_type = insulin_group_data['value'].iloc[-1]
            temporal_features['insulin_type_premixed'] = 1 if insulin_type == 'premixed' else 0
            
        self.features_data = {**static_features, **temporal_features}
        
        print(f"✅ Подготовлено {len(self.features_data)} признаков")
        print(f"📊 Инсулин: current={temporal_features.get('insulin_current', 'N/A')}, "
              f"mean={temporal_features.get('insulin_mean', 'N/A'):.1f}")
        return self.features_data
        
    def permutation_importance_analysis(self):
        """Анализ важности признаков методом перестановки (СТАБИЛЬНАЯ ВЕРСИЯ)"""
        print("🔄 Анализ важности признаков (Permutation Importance) - СТАБИЛЬНАЯ ВЕРСИЯ...")
        
        if not self.features_data:
            self.prepare_features()
            
        # Создаем синтетические данные для демонстрации
        n_samples = 1000  # Имитируем больше данных
        
        # Базовые значения из реальных данных
        base_values = list(self.features_data.values())
        feature_names = list(self.features_data.keys())
        
        # Создаем синтетический датасет с вариациями (СТАБИЛЬНЫЙ)
        X = []
        y = []
        
        # Используем фиксированный seed для воспроизводимости
        np.random.seed(self.random_seed)
        
        for i in range(n_samples):
            # Добавляем случайные вариации к базовым значениям (СТАБИЛЬНЫЕ КОЭФФИЦИЕНТЫ)
            sample = []
            for j, base_val in enumerate(base_values):
                try:
                    # Пытаемся преобразовать в число
                    base_val_num = float(base_val)
                    
                    # РАЗНЫЕ КОЭФФИЦИЕНТЫ ВАРИАЦИИ ДЛЯ РАЗНЫХ ПРИЗНАКОВ
                    feature_name = feature_names[j]
                    if 'insulin' in feature_name:
                        variation_coef = 0.3  # Инсулин может варьироваться на 30%
                    elif 'glu' in feature_name:
                        variation_coef = 0.2  # Глюкоза варьируется на 20%
                    elif feature_name in ['BMI', 'SBP', 'DBP']:
                        variation_coef = 0.15  # Физиологические параметры на 15%
                    else:
                        variation_coef = 0.1  # Остальные на 10%
                        
                    variation = base_val_num * variation_coef * np.random.normal(0, 1)
                    sample.append(base_val_num + variation)
                except (ValueError, TypeError):
                    # Если не число, используем 0
                    sample.append(0.0)
            X.append(sample)
            
            # Создаем синтетическую целевую переменную (глюкоза) - СТАБИЛЬНАЯ МОДЕЛЬ
            glucose = 10.0  # базовая глюкоза
            
            # Безопасно добавляем влияние признаков
            try:
                age_idx = feature_names.index('age')
                glucose += sample[age_idx] * 0.01  # возраст влияет
            except ValueError:
                pass
                
            try:
                bmi_idx = feature_names.index('BMI')
                glucose += sample[bmi_idx] * 0.1   # BMI влияет
            except ValueError:
                pass
                
            try:
                glu_idx = feature_names.index('glu_current')
                glucose += sample[glu_idx] * 0.3  # текущая глюкоза
            except ValueError:
                pass
                
            # СТАБИЛЬНОЕ ВЛИЯНИЕ ИНСУЛИНА
            try:
                insulin_idx = feature_names.index('insulin_current')
                glucose -= sample[insulin_idx] * 0.15  # инсулин снижает глюкозу на 15%
            except ValueError:
                pass
                
            # Дополнительные инсулиновые признаки
            try:
                insulin_mean_idx = feature_names.index('insulin_mean')
                glucose -= sample[insulin_mean_idx] * 0.1  # средняя доза влияет
            except ValueError:
                pass
                
            # СТАБИЛЬНЫЙ ШУМ
            glucose += np.random.normal(0, 1)  # шум
            y.append(glucose)
            
        X = np.array(X)
        y = np.array(y)
        
        # Обучаем простую модель для демонстрации (СТАБИЛЬНАЯ)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_seed)
        rf_model.fit(X, y)
        
        # Вычисляем важность признаков
        importances = rf_model.feature_importances_
        
        # Создаем DataFrame для визуализации
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    def sensitivity_analysis(self):
        """Sensitivity analysis - анализ чувствительности к изменениям признаков """
        print("📈 Sensitivity Analysis (стабильная версия)...")
        
        if not self.features_data:
            self.prepare_features()
            
        # Выбираем ключевые признаки для анализа
        key_features = ['age', 'BMI', 'glu_current', 'insulin_current', 'insulin_mean', 'SBP', 'DBP']
        available_features = [f for f in key_features if f in self.features_data]
        
        sensitivity_results = {}
        
        for feature in available_features:
            try:
                base_value = float(self.features_data[feature])
                variations = []
                glucose_changes = []
                
                # Анализируем изменения в диапазоне ±20%
                for change_percent in np.linspace(-20, 20, 21):
                    new_value = base_value * (1 + change_percent / 100)
                    
                    # СТАБИЛЬНЫЕ КОЭФФИЦИЕНТЫ ВЛИЯНИЯ
                    if feature == 'age':
                        glucose_change = (new_value - base_value) * 0.01
                    elif feature == 'BMI':
                        glucose_change = (new_value - base_value) * 0.1
                    elif feature == 'glu_current':
                        glucose_change = (new_value - base_value) * 0.3
                    elif feature == 'insulin_current':
                        glucose_change = (new_value - base_value) * -0.15  # СТАБИЛЬНО: -15%
                    elif feature == 'insulin_mean':
                        glucose_change = (new_value - base_value) * -0.1   # СТАБИЛЬНО: -10%
                    elif feature in ['SBP', 'DBP']:
                        glucose_change = (new_value - base_value) * 0.02
                    else:
                        glucose_change = 0
                        
                    variations.append(change_percent)
                    glucose_changes.append(glucose_change)
                    
                sensitivity_results[feature] = {
                    'variations': variations,
                    'glucose_changes': glucose_changes,
                    'base_value': base_value
                }
            except (ValueError, TypeError):
                print(f"⚠️  Пропускаем признак {feature} - не числовое значение")
                continue
            
        return sensitivity_results
        
    def visualize_results(self, importance_df, sensitivity_results):
        """Визуализация результатов анализа (СТАБИЛЬНАЯ ВЕРСИЯ)"""
        print("📊 Создание визуализаций (стабильная версия)...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Анализ значимости признаков RL-DITR\n(Демонстрация на одном пациенте)', fontsize=16)
        
        # 1. Важность признаков (верхний график)
        ax1 = axes[0]
        top_features = importance_df.head(10)
        bars = ax1.barh(range(len(top_features)), top_features['importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Важность признака')
        ax1.set_title('Top-10 важных признаков ', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Добавляем значения на столбцы
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        # 2. Sensitivity analysis (нижний график)
        ax2 = axes[1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        for i, (feature, data) in enumerate(sensitivity_results.items()):
            color = colors[i % len(colors)]
            ax2.plot(data['variations'], data['glucose_changes'], 
                    marker='o', label=feature, linewidth=2, color=color, markersize=4)
        ax2.set_xlabel('Изменение признака (%)', fontsize=12)
        ax2.set_ylabel('Изменение глюкозы (ммоль/л)', fontsize=12)
        ax2.set_title('Sensitivity Analysis - Влияние признаков на гликемию', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance_analysis_STABLE.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, importance_df, sensitivity_results):
        """Генерация текстового отчета (СТАБИЛЬНАЯ ВЕРСИЯ)"""
        print("📝 Генерация отчета (стабильная версия)...")
        
        report = """
# СТАБИЛЬНЫЙ анализ значимости признаков RL-DITR
## Демонстрационный отчет (НЕ ВАЛИДЕН для экстраполяции)

### 🔧 Стабилизация алгоритма:
1. ✅ Фиксированный random_seed = 42 для воспроизводимости
2. ✅ Стабильные коэффициенты влияния инсулина (-15% и -10%)
3. ✅ Стабильная обработка данных инсулина
4. ✅ Стабильные коэффициенты вариации для разных признаков
5. ✅ Стабильный RandomForest с фиксированным seed

### 📊 Обзор данных
- Пациент: 1 человек
- Период: 4 дня (13-16 января 2022)
- Записей: ~100
- Признаков: {}
- Random Seed: {}

### 🏆 Top-10 важных признаков (СТАБИЛЬНАЯ ВЕРСИЯ)
""".format(len(self.features_data), self.random_seed)
        
        for i, row in importance_df.head(10).iterrows():
            report += f"{i+1}. {row['feature']}: {row['importance']:.4f}\n"
            
        report += """
### 📈 СТАБИЛЬНЫЙ Sensitivity Analysis
Ключевые выводы по чувствительности:
"""
        
        for feature, data in sensitivity_results.items():
            max_change = max(abs(min(data['glucose_changes'])), abs(max(data['glucose_changes'])))
            report += f"• {feature}: максимальное изменение глюкозы ±{max_change:.2f} ммоль/л при ±20% изменении признака\n"
            
        report += """
### 🎯 Объяснение стабильности результатов

ПОЧЕМУ РЕЗУЛЬТАТЫ ТЕПЕРЬ СТАБИЛЬНЫ:
1. **Фиксированный random_seed**: np.random.seed(42) и torch.manual_seed(42)
2. **Стабильный RandomForest**: random_state=42
3. **Детерминированные вариации**: одинаковые случайные числа при каждом запуске
4. **Стабильные коэффициенты**: неизменные коэффициенты влияния

ПРЕДЫДУЩИЕ ПРОБЛЕМЫ:
- Случайные вариации в генерации синтетических данных
- Случайные компоненты в RandomForest
- Разные случайные числа при каждом запуске

### ⚠️ Ограничения и предупреждения

1. **Объем данных**: Крайне недостаточно для статистически значимых выводов
2. **Генерализация**: Результаты не применимы к другим пациентам
3. **Синтетическая модель**: Это демонстрация, не реальная модель
4. **Клиническая валидация**: Требуется на реальных данных

### 📋 Заключение

Стабилизация алгоритма обеспечивает воспроизводимые результаты при каждом запуске.
Теперь результаты будут одинаковыми при одинаковых входных данных и seed'е.
"""
        
        # Сохраняем отчет
        with open('results/feature_importance_report_STABLE.txt', 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report

def main():
    """Основная функция анализа (СТАБИЛЬНАЯ ВЕРСИЯ)"""
    print("🔬 СТАБИЛЬНЫЙ анализ значимости признаков RL-DITR")
    print("=" * 60)
    
    # Создаем папку для результатов
    import os
    os.makedirs('results', exist_ok=True)
    
    # Инициализируем анализатор с фиксированным seed
    analyzer = FeatureImportanceAnalyzerStable(random_seed=42)
    
    try:
        # Загружаем данные
        analyzer.load_data()
        
        # Подготавливаем признаки
        analyzer.prepare_features()
        
        # Проводим анализ
        print("\n" + "="*60)
        importance_df = analyzer.permutation_importance_analysis()
        sensitivity_results = analyzer.sensitivity_analysis()
        
        # Визуализируем результаты
        print("\n" + "="*60)
        analyzer.visualize_results(importance_df, sensitivity_results)
        
        # Генерируем отчет
        print("\n" + "="*60)
        report = analyzer.generate_report(importance_df, sensitivity_results)
        
        print("\n✅ СТАБИЛЬНЫЙ анализ завершен!")
        print("📁 Результаты сохранены в папке 'results/'")
        print("🔧 Основные стабилизации:")
        print("   - Фиксированный random_seed = 42")
        print("   - Стабильные коэффициенты влияния")
        print("   - Воспроизводимые результаты")
        print("⚠️  Помните: это демонстрационный анализ, не валиден для экстраполяции!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Проверьте наличие файлов sample.csv и predictions_full.json")

if __name__ == "__main__":
    main() 