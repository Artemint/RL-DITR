import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# === Параметры ===
PREDICTIONS_PATH = 'results/predictions_full.json'  # Новый формат вывода
DIABETES_CSV = 'assets/data/diabetes_t1_input.csv'  # Наши данные диабетика
SIMULATION_CSV = 'assets/data/diabetes_t1_input_simulation_period.csv'  # Данные за период симуляции
SAMPLE_CSV = 'assets/data/sample.csv'  # Путь к исходным данным

# === 1. Загрузка результата инференса ===
print('Загрузка результата инференса...')
with open(PREDICTIONS_PATH) as f:
    result = json.load(f)

recommendations = result['recommendations']
glucose_profile = result['glucose_profile']
observed_glucose = result.get('observed_glucose', [])

# === 2. Загрузка реальных данных за период симуляции ===
print('Загрузка реальных данных за период симуляции...')

# Загружаем данные за период симуляции
df_simulation = pd.read_csv(SIMULATION_CSV)
df_simulation['datetime'] = pd.to_datetime(df_simulation['datetime'])

# Получаем реальные данные глюкозы за период симуляции
real_glucose_sim = df_simulation[df_simulation['key_group'] == 'glu'].copy()
real_glucose_sim = real_glucose_sim.sort_values('datetime')

# Получаем реальные дозы инсулина за период симуляции
real_insulin_sim = df_simulation[
    (df_simulation['key_group'] == 'insulin') & 
    (df_simulation['key'] == 'insulin')
].copy()
real_insulin_sim = real_insulin_sim.sort_values('datetime')

# Получаем типы инсулина
real_insulin_types = df_simulation[
    (df_simulation['key_group'] == 'insulin') & 
    (df_simulation['key'] == 'insulin_group')
].copy()
real_insulin_types = real_insulin_types.sort_values('datetime')

print(f"📊 Реальных измерений глюкозы за период симуляции: {len(real_glucose_sim)}")
print(f"📊 Реальных инъекций инсулина за период симуляции: {len(real_insulin_sim)}")

# Анализируем типы инсулина
print(f"\n🔍 Анализ типов инсулина:")
print("Реальные данные:")
for _, row in real_insulin_types.iterrows():
    print(f"  {row['datetime']}: {row['value']}")

# === 3. Загрузка исторических данных ===
# Исторические значения (реальные)
if observed_glucose:
    dates_obs = [step['datetime'] for step in observed_glucose]
    glu_obs = [step['observed_glucose'] for step in observed_glucose]
    # Сортируем по datetime
    obs_tuples = sorted(zip(dates_obs, glu_obs), key=lambda x: pd.to_datetime(x[0]))
    dates_obs, glu_obs = zip(*obs_tuples)
    dates_obs, glu_obs = list(dates_obs), list(glu_obs)
else:
    dates_obs, glu_obs = [], []

# === 4. Подготовка данных для графика ===
# Симуляция (вся симуляция)
dates_glu_sim = [step['datetime'] for step in glucose_profile]
glu_pred_sim = [step['predicted_glucose'] for step in glucose_profile]

# Сортировка симуляции
if dates_glu_sim:
    sim_tuples = sorted(zip(dates_glu_sim, glu_pred_sim), key=lambda x: pd.to_datetime(x[0]))
    dates_glu_sim, glu_pred_sim = zip(*sim_tuples)
    dates_glu_sim, glu_pred_sim = list(dates_glu_sim), list(glu_pred_sim)

# Дозы по симуляции (все рекомендации)
dates_dose_sim = [step['datetime'] for step in recommendations]
doses_sim = [step['dose'] for step in recommendations]
if dates_dose_sim:
    dose_sim_tuples = sorted(zip(dates_dose_sim, doses_sim), key=lambda x: pd.to_datetime(x[0]))
    dates_dose_sim, doses_sim = zip(*dose_sim_tuples)
    dates_dose_sim, doses_sim = list(dates_dose_sim), list(doses_sim)

# === 5. Визуализация ===
fig = plt.figure(figsize=(16, 8))

# График на всю ширину
ax1 = fig.add_subplot(111)

# 1. История: реальные значения глюкозы (до симуляции)
if dates_obs:
    dates_obs_dt = [pd.to_datetime(dt) for dt in dates_obs]
    ax1.plot(dates_obs_dt, glu_obs, marker='s', linestyle='-', color='green', 
             label='Гликемия (реальные данные)', linewidth=2, markersize=6)

# 2. Реальные данные за период симуляции (пунктир)
if not real_glucose_sim.empty:
    real_glu_dates = real_glucose_sim['datetime'].tolist()
    real_glu_values = real_glucose_sim['value'].astype(float).tolist()
    ax1.plot(real_glu_dates, real_glu_values, marker='o', linestyle='--', 
             color='darkgreen', label='Гликемия (реальные данные за период симуляции)', 
             linewidth=2, markersize=6, alpha=0.8)

# 3. Симуляция: прогноз глюкозы
if dates_glu_sim:
    dates_glu_sim_dt = [pd.to_datetime(dt) for dt in dates_glu_sim]
    ax1.plot(dates_glu_sim_dt, glu_pred_sim, marker='o', linestyle='-', 
             color='blue', label='Гликемия (симуляция)', linewidth=2, markersize=6)

# 4. Рекомендуемые дозы инсулина
if dates_dose_sim:
    dates_dose_sim_dt = [pd.to_datetime(dt) for dt in dates_dose_sim]
    # Находим соответствующие значения глюкозы для доз
    dose_glu_values = []
    for dt in dates_dose_sim:
        if dt in dates_glu_sim:
            dose_glu_values.append(glu_pred_sim[dates_glu_sim.index(dt)])
        else:
            dose_glu_values.append(None)
    
    # Фильтруем только валидные точки
    valid_doses = [(dt, dose, glu) for dt, dose, glu in 
                   zip(dates_dose_sim_dt, doses_sim, dose_glu_values) if glu is not None]
    if valid_doses:
        valid_dates, valid_doses_list, valid_glu = zip(*valid_doses)
        ax1.scatter(valid_dates, valid_glu, color='red', marker='x', s=120, 
                   label='Введение инсулина (рекомендация модели)', linewidth=2)
        for dt, dose in zip(valid_dates, valid_doses_list):
            ax1.annotate(f'{dose}', (dt, valid_glu[valid_dates.index(dt)]), 
                        textcoords="offset points", xytext=(0,10), ha='center', 
                        color='red', fontweight='bold')

# Вертикальная линия — граница между историей и симуляцией
if dates_obs and dates_glu_sim:
    start_simulation = pd.to_datetime('2022-01-17 00:00:00')
    ax1.axvline(x=start_simulation, color='gray', linestyle='--', 
                label='Старт симуляции', linewidth=2)

# Добавляем горизонтальные линии для нормальных значений
ax1.axhline(y=3.9, color='lightgray', linestyle=':', alpha=0.7, label='Нижняя граница нормы (3.9)')
ax1.axhline(y=10.0, color='lightgray', linestyle=':', alpha=0.7, label='Верхняя граница нормы (10.0)')

ax1.set_xlabel('Дата и время', fontsize=12)
ax1.set_ylabel('Глюкоза (ммоль/л)', fontsize=12)
ax1.set_title('Гликемический профиль: история, симуляция и реальные данные', 
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=3)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# === Создание отдельного файла с таблицей ===
if dates_dose_sim and not real_insulin_sim.empty:
    # Подготавливаем данные для таблицы
    table_data = []

    # Рекомендуемые дозы (с правильными типами инсулина)
    for i, (rec_date, rec_dose) in enumerate(zip(dates_dose_sim, doses_sim)):
        rec_dt = pd.to_datetime(rec_date)
        
        # Получаем тип инсулина из рекомендаций
        insulin_type = "Короткий"
        for rec in recommendations:
            if rec['datetime'] == rec_date:
                if rec.get('insulin_type') == 'long':
                    insulin_type = "Длинный"
                elif rec.get('insulin_type') == 'medium':
                    insulin_type = "Средний"
                break
        
        table_data.append({
            'Время': rec_dt.strftime('%d.%m %H:%M'),
            'Тип': f'Рекомендация ({insulin_type})',
            'Доза': f"{rec_dose} ед.",
            'Дата': rec_dt.date()
        })

    # Реальные дозы
    for i, (real_date, real_dose) in enumerate(zip(real_insulin_sim['datetime'], real_insulin_sim['value'])):
        real_dt = pd.to_datetime(real_date)
        # Определяем тип инсулина
        insulin_type = "Короткий"
        nearby_type = real_insulin_types[
            (real_insulin_types['datetime'] == real_date)
        ]
        if not nearby_type.empty:
            insulin_type_value = nearby_type.iloc[0]['value']
            if insulin_type_value == 'long':
                insulin_type = "Длинный"

        table_data.append({
            'Время': real_dt.strftime('%d.%m %H:%M'),
            'Тип': f'Реальный ({insulin_type})',
            'Доза': f"{real_dose} ед.",
            'Дата': real_dt.date()
        })

    # Сортируем по времени
    table_data.sort(key=lambda x: pd.to_datetime(x['Время'], format='%d.%m %H:%M'))

    # Создаем DataFrame и сохраняем в CSV
    df_table = pd.DataFrame(table_data)
    df_table.to_csv('results/insulin_comparison_table.csv', index=False, encoding='utf-8-sig')
    print(f"📊 Таблица сохранена в: results/insulin_comparison_table.csv")

# Сохраняем график
plt.savefig('results/glucose_profile_visualization_improved.png', dpi=300, bbox_inches='tight')
plt.show()

print('Готово!')
print('📊 График сохранен в: results/glucose_profile_visualization_improved.png')

# === 6. Анализ сравнения ===
print('\n📊 Анализ сравнения:')
print('=' * 50)

if not real_insulin_sim.empty and dates_dose_sim:
    print('\n🔍 Сравнение доз инсулина:')
    print('Рекомендации модели vs Реальные данные')
    print('-' * 40)
    
    # Создаем DataFrame для сравнения
    comparison_data = []
    
    for i, (rec_date, rec_dose) in enumerate(zip(dates_dose_sim, doses_sim)):
        rec_dt = pd.to_datetime(rec_date)
        
        # Ищем ближайшую реальную дозу в пределах 2 часов
        time_diff = pd.Timedelta(hours=2)
        nearby_real = real_insulin_sim[
            (real_insulin_sim['datetime'] >= rec_dt - time_diff) &
            (real_insulin_sim['datetime'] <= rec_dt + time_diff)
        ]
        
        if not nearby_real.empty:
            closest_idx = (nearby_real['datetime'] - rec_dt).abs().idxmin()
            real_dose = nearby_real.loc[closest_idx, 'value']
            real_time = nearby_real.loc[closest_idx, 'datetime']
            time_diff_minutes = abs((real_time - rec_dt).total_seconds() / 60)
            
            # Определяем тип реального инсулина
            nearby_type = real_insulin_types[
                (real_insulin_types['datetime'] == real_time)
            ]
            insulin_type = "Короткий"
            if not nearby_type.empty:
                insulin_type_value = nearby_type.iloc[0]['value']
                if insulin_type_value == 'long':
                    insulin_type = "Длинный"
            
            # Получаем тип инсулина из рекомендаций
            model_insulin_type = "Короткий"
            for rec in recommendations:
                if rec['datetime'] == rec_date:
                    if rec.get('insulin_type') == 'long':
                        model_insulin_type = "Длинный"
                    elif rec.get('insulin_type') == 'medium':
                        model_insulin_type = "Средний"
                    break
            
            comparison_data.append({
                'Рекомендация': f"{rec_dt.strftime('%H:%M')} - {rec_dose} ед. ({model_insulin_type})",
                'Реальная': f"{real_time.strftime('%H:%M')} - {real_dose} ед. ({insulin_type})",
                'Разница доз': rec_dose - float(real_dose),
                'Время (мин)': time_diff_minutes
            })
            
            # Получаем тип инсулина из рекомендаций
            model_insulin_type = "Короткий"
            for rec in recommendations:
                if rec['datetime'] == rec_date:
                    if rec.get('insulin_type') == 'long':
                        model_insulin_type = "Длинный"
                    elif rec.get('insulin_type') == 'medium':
                        model_insulin_type = "Средний"
                    break
            
            print(f"Модель: {rec_dt.strftime('%H:%M')} - {rec_dose} ед. ({model_insulin_type})")
            print(f"Реально: {real_time.strftime('%H:%M')} - {real_dose} ед. ({insulin_type})")
            print(f"Разница: {rec_dose - float(real_dose)} ед. (время: {time_diff_minutes:.0f} мин)")
            print()
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        avg_dose_diff = df_comparison['Разница доз'].mean()
        avg_time_diff = df_comparison['Время (мин)'].mean()
        
        print(f"📈 Статистика сравнения:")
        print(f"Средняя разница в дозах: {avg_dose_diff:.1f} ед.")
        print(f"Средняя разница во времени: {avg_time_diff:.0f} мин")
        print(f"Количество совпадений: {len(comparison_data)}")

# Сохраняем отчет о сравнении
if 'comparison_data' in locals() and comparison_data:
    comparison_report = {
        'comparison_data': comparison_data,
        'statistics': {
            'avg_dose_diff': avg_dose_diff,
            'avg_time_diff': avg_time_diff,
            'total_comparisons': len(comparison_data)
        }
    }
    
    with open('results/comparison_report_improved.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n📄 Отчет о сравнении сохранен в: results/comparison_report_improved.json") 