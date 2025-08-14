#!/usr/bin/env python3


import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# === Параметры ===
DIABETES_CSV = 'assets/data/diabetes_t1_input.csv'
SIMULATION_CSV = 'assets/data/diabetes_t1_input_simulation_period.csv'
VALIDATION_PERIOD = '2022-01-17'
VALIDATION_DAYS = 2

class ModelValidatorFixed:
    """Исправленная версия валидатора модели RL-DITR"""
    
    def __init__(self):
        self.model = None
        self.df_validation = None
        self.real_glucose = None
        self.real_insulin = None
        self.simulated_glucose = None
        self.validation_metrics = {}
        
    def load_model(self):
        """Загружает модель RL-DITR"""
        print("🔧 Загрузка модели...")
        
        try:
            from ts.arm import Model
            model_dir = Path('assets/models/weights')
            df_meta_path = 'assets/models/features.csv'
            
            self.model = Model(model_dir, df_meta_path, beam_size=1)
            print("✅ Модель загружена успешно")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def load_validation_data(self):
        """Загружает данные для валидации"""
        print("📊 Загрузка данных для валидации...")
        
        # Загружаем данные за период симуляции
        df_simulation = pd.read_csv(SIMULATION_CSV)
        df_simulation['datetime'] = pd.to_datetime(df_simulation['datetime'])
        
        # Фильтруем данные за период валидации
        start_date = pd.to_datetime(VALIDATION_PERIOD)
        end_date = start_date + timedelta(days=VALIDATION_DAYS)
        
        self.df_validation = df_simulation[
            (df_simulation['datetime'] >= start_date) &
            (df_simulation['datetime'] < end_date)
        ].copy()
        
        # Получаем реальные данные глюкозы
        self.real_glucose = self.df_validation[
            self.df_validation['key_group'] == 'glu'
        ].copy().sort_values('datetime')
        
        # Получаем реальные дозы инсулина
        self.real_insulin = self.df_validation[
            (self.df_validation['key_group'] == 'insulin') & 
            (self.df_validation['key'] == 'insulin')
        ].copy().sort_values('datetime')
        
        # Получаем типы инсулина
        self.real_insulin_types = self.df_validation[
            (self.df_validation['key_group'] == 'insulin') & 
            (self.df_validation['key'] == 'insulin_group')
        ].copy().sort_values('datetime')
        
        print(f"📊 Загружено {len(self.real_glucose)} измерений глюкозы")
        print(f"📊 Загружено {len(self.real_insulin)} инъекций инсулина")
        
        # Отладочная информация
        print(f"📅 Период валидации: {start_date} - {end_date}")
        print(f"📊 Диапазон реальной глюкозы: {self.real_glucose['value'].astype(float).min():.1f} - {self.real_glucose['value'].astype(float).max():.1f} ммоль/л")
        
        return len(self.real_glucose) > 0 and len(self.real_insulin) > 0
    
    def create_validation_input(self):
        """Создает входные данные для валидации с реальными дозами инсулина"""
        print("🔧 Создание входных данных для валидации...")
        
        # Создаем копию исходных данных
        df_input = pd.read_csv(DIABETES_CSV)
        df_input['datetime'] = pd.to_datetime(df_input['datetime'])
        
        # Заменяем дозы инсулина на реальные из периода валидации
        validation_start = pd.to_datetime(VALIDATION_PERIOD)
        
        # Удаляем старые дозы инсулина за период валидации
        df_input = df_input[
            ~((df_input['key_group'] == 'insulin') & 
              (df_input['datetime'] >= validation_start))
        ]
        
        # Добавляем реальные дозы инсулина
        real_insulin_data = []
        for _, row in self.real_insulin.iterrows():
            real_insulin_data.append({
                'datetime': row['datetime'],
                'key_group': 'insulin',
                'key': 'insulin',
                'value': row['value'],
                'key_type': 'cont'
            })
            
            # Добавляем тип инсулина
            nearby_type = self.real_insulin_types[
                self.real_insulin_types['datetime'] == row['datetime']
            ]
            if not nearby_type.empty:
                insulin_type = nearby_type.iloc[0]['value']
                real_insulin_data.append({
                    'datetime': row['datetime'],
                    'key_group': 'insulin',
                    'key': 'insulin_group',
                    'value': insulin_type,
                    'key_type': 'cat2'
                })
        
        # Добавляем реальные данные в конец
        df_real_insulin = pd.DataFrame(real_insulin_data)
        df_input = pd.concat([df_input, df_real_insulin], ignore_index=True)
        
        # Сортируем по времени
        df_input = df_input.sort_values('datetime')
        
        # Сохраняем файл для валидации
        validation_input_path = 'assets/data/diabetes_t1_input_validation_fixed.csv'
        df_input.to_csv(validation_input_path, index=False)
        
        print(f"✅ Входные данные для валидации сохранены: {validation_input_path}")
        print(f"📊 Всего записей в файле: {len(df_input)}")
        print(f"📊 Инсулиновых записей: {len(df_input[df_input['key_group'] == 'insulin'])}")
        
        return validation_input_path
    
    def run_validation_simulation(self, input_path):
        """Запускает симуляцию с реальными дозами инсулина"""
        print("🔄 Запуск симуляции валидации...")
        
        try:
            # Запускаем инференс с реальными дозами
            result = self.model.predict(
                df=input_path,
                scheme='validation',
                start_time=VALIDATION_PERIOD,
                days=VALIDATION_DAYS
            )
            
            # Отладочная информация о результате
            print(f"📊 Результат содержит:")
            print(f"   - Рекомендации: {len(result.get('recommendations', []))}")
            print(f"   - Профиль глюкозы: {len(result.get('glucose_profile', []))}")
            print(f"   - Наблюдаемая глюкоза: {len(result.get('observed_glucose', []))}")
            
            # Извлекаем симулированную глюкозу
            self.simulated_glucose = []
            glucose_profile = result.get('glucose_profile', [])
            
            for step in glucose_profile:
                if isinstance(step, dict) and 'predicted_glucose' in step:
                    self.simulated_glucose.append({
                        'datetime': step['datetime'],
                        'predicted_glucose': step['predicted_glucose']
                    })
            
            print(f"✅ Симуляция завершена: {len(self.simulated_glucose)} точек")
            
            # Отладочная информация о симулированной глюкозе
            if self.simulated_glucose:
                sim_values = [s['predicted_glucose'] for s in self.simulated_glucose]
                print(f"📊 Диапазон симулированной глюкозы: {min(sim_values):.1f} - {max(sim_values):.1f} ммоль/л")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка симуляции: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def align_data_for_comparison(self):
        """Выравнивает данные для сравнения по времени"""
        print("🔧 Выравнивание данных для сравнения...")
        
        if not self.simulated_glucose:
            print("❌ Нет симулированных данных")
            return False
        
        # Создаем DataFrame с симулированными данными
        df_sim = pd.DataFrame(self.simulated_glucose)
        df_sim['datetime'] = pd.to_datetime(df_sim['datetime'])
        
        # Создаем DataFrame с реальными данными
        df_real = self.real_glucose[['datetime', 'value']].copy()
        df_real.columns = ['datetime', 'real_glucose']
        df_real['real_glucose'] = df_real['real_glucose'].astype(float)
        
        print(f"📊 Симулированных точек: {len(df_sim)}")
        print(f"📊 Реальных точек: {len(df_real)}")
        
        # Объединяем данные по времени (ближайшее совпадение)
        merged_data = []
        
        for _, sim_row in df_sim.iterrows():
            sim_time = sim_row['datetime']
            sim_glucose = sim_row['predicted_glucose']
            
            # Ищем ближайшее реальное измерение в пределах 30 минут
            time_diff = pd.Timedelta(minutes=30)
            nearby_real = df_real[
                (df_real['datetime'] >= sim_time - time_diff) &
                (df_real['datetime'] <= sim_time + time_diff)
            ]
            
            if not nearby_real.empty:
                closest_idx = (nearby_real['datetime'] - sim_time).abs().idxmin()
                real_glucose = nearby_real.loc[closest_idx, 'real_glucose']
                real_time = nearby_real.loc[closest_idx, 'datetime']
                time_diff_minutes = abs((real_time - sim_time).total_seconds() / 60)
                
                merged_data.append({
                    'datetime': sim_time,
                    'simulated_glucose': sim_glucose,
                    'real_glucose': real_glucose,
                    'time_diff_minutes': time_diff_minutes
                })
                
                print(f"🔗 Сопоставление: {sim_time} (сим: {sim_glucose:.1f}) - {real_time} (реал: {real_glucose:.1f}) [разница: {time_diff_minutes:.0f} мин]")
        
        self.comparison_data = pd.DataFrame(merged_data)
        
        print(f"✅ Выравнено {len(self.comparison_data)} точек для сравнения")
        
        if len(self.comparison_data) > 0:
            print(f"📊 Диапазон реальной глюкозы в сравнении: {self.comparison_data['real_glucose'].min():.1f} - {self.comparison_data['real_glucose'].max():.1f}")
            print(f"📊 Диапазон симулированной глюкозы: {self.comparison_data['simulated_glucose'].min():.1f} - {self.comparison_data['simulated_glucose'].max():.1f}")
        
        return len(self.comparison_data) > 0
    
    def calculate_validation_metrics(self):
        """Вычисляет метрики валидации"""
        print("📊 Вычисление метрик валидации...")
        
        if self.comparison_data.empty:
            print("❌ Нет данных для сравнения")
            return
        
        # Проверяем данные на корректность
        print(f"📊 Проверка данных:")
        print(f"   - Реальная глюкоза: {self.comparison_data['real_glucose'].describe()}")
        print(f"   - Симулированная глюкоза: {self.comparison_data['simulated_glucose'].describe()}")
        
        # Основные метрики
        mae = mean_absolute_error(
            self.comparison_data['real_glucose'], 
            self.comparison_data['simulated_glucose']
        )
        
        mse = mean_squared_error(
            self.comparison_data['real_glucose'], 
            self.comparison_data['simulated_glucose']
        )
        
        rmse = np.sqrt(mse)
        
        r2 = r2_score(
            self.comparison_data['real_glucose'], 
            self.comparison_data['simulated_glucose']
        )
        
        # Дополнительные метрики
        mean_absolute_percentage_error = np.mean(
            np.abs((self.comparison_data['real_glucose'] - self.comparison_data['simulated_glucose']) / 
                   self.comparison_data['real_glucose']) * 100
        )
        
        # Анализ по диапазонам глюкозы
        self.comparison_data['glucose_range'] = pd.cut(
            self.comparison_data['real_glucose'],
            bins=[0, 3.9, 10.0, 15.0, 30.0],
            labels=['Гипогликемия', 'Норма', 'Гипергликемия', 'Высокая гипергликемия']
        )
        
        range_metrics = {}
        for range_name in self.comparison_data['glucose_range'].unique():
            if pd.notna(range_name):
                range_data = self.comparison_data[
                    self.comparison_data['glucose_range'] == range_name
                ]
                if len(range_data) > 0:
                    range_mae = mean_absolute_error(
                        range_data['real_glucose'], 
                        range_data['simulated_glucose']
                    )
                    range_metrics[range_name] = {
                        'count': len(range_data),
                        'mae': range_mae
                    }
        
        self.validation_metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mean_absolute_percentage_error,
            'total_points': len(self.comparison_data),
            'range_metrics': range_metrics
        }
        
        print("✅ Метрики вычислены")
    
    def visualize_validation_results(self):
        """Визуализирует результаты валидации"""
        print("📊 Создание визуализации...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # График 1: Сравнение временных рядов
        ax1.plot(self.comparison_data['datetime'], self.comparison_data['real_glucose'], 
                'o-', color='blue', label='Реальная глюкоза', linewidth=2, markersize=6)
        ax1.plot(self.comparison_data['datetime'], self.comparison_data['simulated_glucose'], 
                's-', color='red', label='Симулированная глюкоза', linewidth=2, markersize=6)
        ax1.axhline(y=3.9, color='lightgray', linestyle=':', alpha=0.7, label='Нижняя граница нормы')
        ax1.axhline(y=10.0, color='lightgray', linestyle=':', alpha=0.7, label='Верхняя граница нормы')
        ax1.set_ylabel('Глюкоза (ммоль/л)')
        ax1.set_title('Сравнение реальной и симулированной глюкозы')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # График 2: Диаграмма рассеяния
        ax2.scatter(self.comparison_data['real_glucose'], self.comparison_data['simulated_glucose'], 
                   alpha=0.6, color='green')
        
        # Линия идеального соответствия
        min_val = min(self.comparison_data['real_glucose'].min(), self.comparison_data['simulated_glucose'].min())
        max_val = max(self.comparison_data['real_glucose'].max(), self.comparison_data['simulated_glucose'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Идеальное соответствие')
        
        ax2.set_xlabel('Реальная глюкоза (ммоль/л)')
        ax2.set_ylabel('Симулированная глюкоза (ммоль/л)')
        ax2.set_title(f'Диаграмма рассеяния (R² = {self.validation_metrics["r2"]:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Распределение ошибок
        errors = self.comparison_data['simulated_glucose'] - self.comparison_data['real_glucose']
        ax3.hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Нет ошибки')
        ax3.set_xlabel('Ошибка (ммоль/л)')
        ax3.set_ylabel('Частота')
        ax3.set_title(f'Распределение ошибок (MAE = {self.validation_metrics["mae"]:.2f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # График 4: Метрики по диапазонам глюкозы
        if self.validation_metrics['range_metrics']:
            ranges = list(self.validation_metrics['range_metrics'].keys())
            maes = [self.validation_metrics['range_metrics'][r]['mae'] for r in ranges]
            counts = [self.validation_metrics['range_metrics'][r]['count'] for r in ranges]
            
            bars = ax4.bar(ranges, maes, color=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow'])
            ax4.set_xlabel('Диапазон глюкозы')
            ax4.set_ylabel('MAE (ммоль/л)')
            ax4.set_title('Точность по диапазонам глюкозы')
            ax4.grid(True, alpha=0.3)
            
            # Добавляем количество точек на столбцы
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/validation_results_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Визуализация сохранена: results/validation_results_fixed.png")
    
    def print_validation_report(self):
        """Выводит отчет о валидации"""
        print("\n" + "="*60)
        print("📊 ОТЧЕТ О ВАЛИДАЦИИ МОДЕЛИ RL-DITR (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
        print("="*60)
        
        print(f"\n📈 Основные метрики:")
        print(f"   MAE (средняя абсолютная ошибка): {self.validation_metrics['mae']:.2f} ммоль/л")
        print(f"   RMSE (среднеквадратичная ошибка): {self.validation_metrics['rmse']:.2f} ммоль/л")
        print(f"   R² (коэффициент детерминации): {self.validation_metrics['r2']:.3f}")
        print(f"   MAPE (средняя абсолютная процентная ошибка): {self.validation_metrics['mape']:.1f}%")
        print(f"   Количество точек сравнения: {self.validation_metrics['total_points']}")
        
        print(f"\n📊 Анализ по диапазонам глюкозы:")
        for range_name, metrics in self.validation_metrics['range_metrics'].items():
            print(f"   {range_name}: MAE = {metrics['mae']:.2f} ммоль/л (n={metrics['count']})")
        
        print(f"\n💡 Интерпретация:")
        if self.validation_metrics['r2'] > 0.8:
            print("   ✅ Отличная точность модели")
        elif self.validation_metrics['r2'] > 0.6:
            print("   ✅ Хорошая точность модели")
        elif self.validation_metrics['r2'] > 0.4:
            print("   ⚠️  Удовлетворительная точность модели")
        else:
            print("   ❌ Низкая точность модели")
        
        print(f"\n📄 Полный отчет сохранен в: results/validation_report_fixed.json")
    
    def save_validation_report(self):
        """Сохраняет полный отчет о валидации"""
        report = {
            'validation_period': {
                'start_date': VALIDATION_PERIOD,
                'days': VALIDATION_DAYS
            },
            'data_summary': {
                'real_glucose_points': len(self.real_glucose),
                'real_insulin_injections': len(self.real_insulin),
                'comparison_points': len(self.comparison_data)
            },
            'metrics': self.validation_metrics,
            'comparison_data': self.comparison_data.to_dict('records')
        }
        
        with open('results/validation_report_fixed.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    
    def run_full_validation(self):
        """Запускает полную валидацию модели"""
        print("🚀 Запуск полной валидации модели RL-DITR (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
        print("="*60)
        
        # 1. Загрузка модели
        if not self.load_model():
            return False
        
        # 2. Загрузка данных
        if not self.load_validation_data():
            print("❌ Недостаточно данных для валидации")
            return False
        
        # 3. Создание входных данных
        input_path = self.create_validation_input()
        
        # 4. Запуск симуляции
        if not self.run_validation_simulation(input_path):
            return False
        
        # 5. Выравнивание данных
        if not self.align_data_for_comparison():
            print("❌ Не удалось выровнять данные для сравнения")
            return False
        
        # 6. Вычисление метрик
        self.calculate_validation_metrics()
        
        # 7. Визуализация
        self.visualize_validation_results()
        
        # 8. Отчет
        self.print_validation_report()
        self.save_validation_report()
        
        print("\n✅ Валидация завершена успешно!")
        return True

def main():
    """Основная функция"""
    validator = ModelValidatorFixed()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎉 Валидация модели выполнена успешно!")
        print("📊 Результаты сохранены в папке results/")
    else:
        print("\n❌ Валидация завершилась с ошибками")

if __name__ == '__main__':
    main() 