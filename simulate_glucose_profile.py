import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# === Параметры ===
PREDICTIONS_PATH = 'results/predictions_full.json'  # Новый формат вывода
SAMPLE_CSV = 'assets/data/sample.csv'  # Путь к исходным данным

# === 1. Загрузка результата инференса ===
print('Загрузка результата инференса...')
with open(PREDICTIONS_PATH) as f:
    result = json.load(f)

recommendations = result['recommendations']
glucose_profile = result['glucose_profile']
observed_glucose = result.get('observed_glucose', [])

# === 1b. Загрузка фактических доз инсулина из sample.csv ===
df_sample = pd.read_csv(SAMPLE_CSV)
df_sample['datetime'] = pd.to_datetime(df_sample['datetime'])
# Фильтруем только строки с инсулином
df_insulin = df_sample[(df_sample['key_group'] == 'insulin') & (df_sample['key'] == 'insulin')]
# Получаем временной ряд доз
insulin_dates = df_insulin['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
insulin_doses = df_insulin['value'].astype(float).tolist()

# === 2. Подготовка данных для графика ===
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

# Фактические введения инсулина (история)
if insulin_dates:
    insulin_tuples = sorted(zip(insulin_dates, insulin_doses), key=lambda x: pd.to_datetime(x[0]))
    insulin_dates, insulin_doses = zip(*insulin_tuples)
    insulin_dates, insulin_doses = list(insulin_dates), list(insulin_doses)

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

# === 3. Визуализация ===
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))

# 1. История: реальные значения глюкозы
if dates_obs:
    dates_obs_dt = [pd.to_datetime(dt) for dt in dates_obs]
    plt.plot(dates_obs_dt, glu_obs, marker='s', linestyle='-', color='green', label='Гликемия (реальные данные)')

# 2. Фактические введения инсулина (история)
if insulin_dates:
    insulin_dates_dt = [pd.to_datetime(dt) for dt in insulin_dates]
    # Для совпадающих дат с историей глюкозы ищем индекс для аннотации
    y_insulin = []
    for dt in insulin_dates:
        if dt in dates_obs:
            y_insulin.append(glu_obs[dates_obs.index(dt)])
        else:
            y_insulin.append(None)
    plt.scatter(insulin_dates_dt, y_insulin, color='orange', label='Введение инсулина (история)', marker='D', s=80)
    for dt, dose, y in zip(insulin_dates_dt, insulin_doses, y_insulin):
        if y is not None:
            plt.annotate(f'{int(dose)}', (dt, y), textcoords="offset points", xytext=(0,10), ha='center', color='black')

# 3. Симуляция: прогноз глюкозы и рекомендованные дозы
if dates_glu_sim:
    # Конвертируем строки в datetime для правильного отображения на оси X
    dates_glu_sim_dt = [pd.to_datetime(dt) for dt in dates_glu_sim]
    plt.plot(dates_glu_sim_dt, glu_pred_sim, marker='o', linestyle='-', color='blue', label='Гликемия (симуляция)')
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
        valid_doses = [(dt, dose, glu) for dt, dose, glu in zip(dates_dose_sim_dt, doses_sim, dose_glu_values) if glu is not None]
        if valid_doses:
            valid_dates, valid_doses_list, valid_glu = zip(*valid_doses)
            plt.scatter(valid_dates, valid_glu, color='red', label='Введение инсулина (рекомендация)', marker='x', s=100)
            for dt, dose in zip(valid_dates, valid_doses_list):
                plt.annotate(f'{dose}', (dt, valid_glu[valid_dates.index(dt)]), textcoords="offset points", xytext=(0,10), ha='center', color='red')

# Вертикальная линия — граница между историей и симуляцией
if dates_obs and dates_glu_sim:
    # Старт симуляции с начала 16 января
    start_simulation = pd.to_datetime('2022-01-16 00:00:00')
    plt.axvline(x=start_simulation, color='gray', linestyle='--', label='Старт симуляции')

plt.xlabel('Время')
plt.ylabel('Глюкоза (ммоль/л)')
plt.title('Гликемический профиль: история и симуляция RL-DITR')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print('Готово!') 