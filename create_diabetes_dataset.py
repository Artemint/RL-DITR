import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def create_sparse_4day_with_insulin():
    """Создает разряженный 4-дневный датасет с инсулином"""
    
    print("🔧 Создание разряженного 4-дневного датасета с инсулином...")
    
    # Загружаем данные пациента
    patient_data = load_patient_data()
    
    # Загружаем данные mySugr
    mysugr_df = pd.read_csv('assets/data/D_1_type/mysugr_data/2022_01_09-2022_04_25_export.csv')
    mysugr_df['datetime'] = pd.to_datetime(mysugr_df['Date'] + ' ' + mysugr_df['Time'])
    
    # Загружаем данные reader
    reader_files = [
        'assets/data/D_1_type/reader_data/2022_04_20-2022_04_25_export.csv',
        'assets/data/D_1_type/reader_data/2022_04_03-2022_04_20_export.csv',
        'assets/data/D_1_type/reader_data/2022_03_19-2022_04_03_export.csv',
        'assets/data/D_1_type/reader_data/2022_03_05-2022_03_19_export.csv',
        'assets/data/D_1_type/reader_data/2022_02_16-2022_03_05_export.csv',
        'assets/data/D_1_type/reader_data/2022_02_06-2022_02_16_export.csv',
        'assets/data/D_1_type/reader_data/2022_01_26-2022_02_06_export.csv',
        'assets/data/D_1_type/reader_data/2022_01_15-2022_01_26_export.csv',
        'assets/data/D_1_type/reader_data/2022_01_06-2022_01_15_export.csv'
    ]
    
    all_reader_data = []
    for file in reader_files:
        if os.path.exists(file):
            df = pd.read_csv(file, sep='\t')
            df['datetime'] = pd.to_datetime(df['Time'])
            all_reader_data.append(df)
    
    reader_df = pd.concat(all_reader_data, ignore_index=True) if all_reader_data else None
    
    # Находим 4 непрерывных дня с инсулином
    # Из анализа: 2022-01-13, 2022-01-14, 2022-01-15, 2022-01-16
    start_date = pd.to_datetime('2022-01-13')
    end_date = pd.to_datetime('2022-01-16')
    
    print(f"📅 Период: {start_date.date()} - {end_date.date()}")
    
    # Создаем входной файл с правильной структурой
    input_data = []
    row_id = 0
    
    # 1. ИНДИВИДУАЛЬНЫЕ ДАННЫЕ ПАЦИЕНТА (в самом начале)
    base_date = start_date.replace(hour=0, minute=0)  # Начало первого дня
    
    # Возраст
    input_data.append({
        'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
        'key_type': 'cont',
        'key_group': 'base',
        'key': 'age',
        'value': 35
    })
    row_id += 1
    
    # Пол
    input_data.append({
        'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
        'key_type': 'cat2(M,F)',
        'key_group': 'base',
        'key': 'gender',
        'value': 'M'
    })
    row_id += 1
    
    # Давление
    input_data.append({
        'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
        'key_type': 'cont',
        'key_group': 'base',
        'key': 'DBP',
        'value': 75
    })
    row_id += 1
    
    input_data.append({
        'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
        'key_type': 'cont',
        'key_group': 'base',
        'key': 'SBP',
        'value': 120
    })
    row_id += 1
    
    # Частота дыхания
    input_data.append({
        'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
        'key_type': 'cont',
        'key_group': 'base',
        'key': 'RR',
        'value': 18
    })
    row_id += 1
    
    # Рост и вес из patient_data
    if 'height_cm' in patient_data:
        input_data.append({
            'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
            'key_type': 'cont',
            'key_group': 'base',
            'key': 'height',
            'value': patient_data['height_cm']
        })
        row_id += 1
    
    if 'weight_kg' in patient_data:
        input_data.append({
            'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
            'key_type': 'cont',
            'key_group': 'base',
            'key': 'weight',
            'value': patient_data['weight_kg']
        })
        row_id += 1
    
    # BMI
    if 'height_cm' in patient_data and 'weight_kg' in patient_data:
        height_m = patient_data['height_cm'] / 100
        bmi = patient_data['weight_kg'] / (height_m ** 2)
        input_data.append({
            'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
            'key_type': 'cont',
            'key_group': 'base',
            'key': 'BMI',
            'value': round(bmi, 2)
        })
        row_id += 1
    
    # 2. ДИАГНОЗ (сразу после базовых данных)
    input_data.append({
        'datetime': base_date.strftime('%Y-%m-%d %H:%M'),
        'key_type': 'cat2',
        'key_group': 'diag',
        'key': 'Type 1 diabetes',
        'value': 1
    })
    row_id += 1
    
    # 3. ВРЕМЕННОЙ РЯД (разряженные данные глюкозы и инсулина)
    
    # Ключевые временные точки для разряженности (не более 20 в день)
    key_times = [
        '06:00', '08:00', '10:00', '12:00', '14:00', 
        '16:00', '18:00', '20:00', '22:00', '00:00'
    ]
    
    # Обрабатываем каждый день
    for day_offset in range(4):
        current_date = start_date + timedelta(days=day_offset)
        day_str = current_date.strftime('%Y-%m-%d')
        
        print(f"\n📅 Обработка дня: {day_str}")
        
        # Глюкоза из reader
        if reader_df is not None:
            day_reader = reader_df[
                (reader_df['datetime'].dt.date == current_date.date()) & 
                (reader_df['Scan Glucose (mmol/L)'].notna())
            ].copy()
            
            # Убираем дубликаты
            day_reader = day_reader.drop_duplicates(subset=['datetime', 'Scan Glucose (mmol/L)'])
            
            print(f"  📊 Измерений глюкозы: {len(day_reader)}")
            
            # Выбираем разряженные точки (не более 10 в день)
            selected_glucose = []
            
            for time_str in key_times:
                # Ищем ближайшее измерение к ключевому времени
                time_obj = datetime.strptime(time_str, '%H:%M').time()
                target_datetime = current_date.replace(hour=time_obj.hour, minute=time_obj.minute)
                
                # Ищем ближайшее измерение в пределах 2 часов
                time_diff = pd.Timedelta(hours=2)
                nearby_measurements = day_reader[
                    (day_reader['datetime'] >= target_datetime - time_diff) &
                    (day_reader['datetime'] <= target_datetime + time_diff)
                ]
                
                if len(nearby_measurements) > 0:
                    # Берем ближайшее измерение
                    closest_idx = (nearby_measurements['datetime'] - target_datetime).abs().idxmin()
                    closest_measurement = nearby_measurements.loc[closest_idx]
                    selected_glucose.append(closest_measurement)
            
            # Добавляем выбранные измерения глюкозы
            for row in selected_glucose:
                input_data.append({
                    'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                    'key_type': 'cont',
                    'key_group': 'glu',
                    'key': 'glu',
                    'value': row['Scan Glucose (mmol/L)']
                })
                row_id += 1
        
        # Инсулин из mySugr
        if mysugr_df is not None:
            day_mysugr = mysugr_df[
                (mysugr_df['datetime'].dt.date == current_date.date())
            ]
            
            print(f"  📊 Записей mySugr: {len(day_mysugr)}")
            
            # Добавляем все инъекции инсулина для этого дня
            for _, row in day_mysugr.iterrows():
                # Определяем тип инсулина
                insulin_type = 'rapid'
                if pd.notna(row['Basal Injection Units']):
                    insulin_type = 'long'
                elif pd.notna(row['Insulin (Meal)']):
                    insulin_type = 'rapid'
                elif pd.notna(row['Insulin (Correction)']):
                    insulin_type = 'rapid'
                
                # Добавляем дозу инсулина
                if pd.notna(row['Basal Injection Units']):
                    input_data.append({
                        'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'key_type': 'cat2',
                        'key_group': 'insulin',
                        'key': 'insulin',
                        'value': row['Basal Injection Units']
                    })
                    row_id += 1
                    
                    input_data.append({
                        'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'key_type': 'cat6(na,medium,short,rapid,long,premixed)',
                        'key_group': 'insulin',
                        'key': 'insulin_group',
                        'value': insulin_type
                    })
                    row_id += 1
                
                if pd.notna(row['Insulin (Meal)']):
                    input_data.append({
                        'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'key_type': 'cat2',
                        'key_group': 'insulin',
                        'key': 'insulin',
                        'value': row['Insulin (Meal)']
                    })
                    row_id += 1
                    
                    input_data.append({
                        'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'key_type': 'cat6(na,medium,short,rapid,long,premixed)',
                        'key_group': 'insulin',
                        'key': 'insulin_group',
                        'value': insulin_type
                    })
                    row_id += 1
                
                if pd.notna(row['Insulin (Correction)']):
                    input_data.append({
                        'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'key_type': 'cat2',
                        'key_group': 'insulin',
                        'key': 'insulin',
                        'value': row['Insulin (Correction)']
                    })
                    row_id += 1
                    
                    input_data.append({
                        'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'key_type': 'cat6(na,medium,short,rapid,long,premixed)',
                        'key_group': 'insulin',
                        'key': 'insulin_group',
                        'value': insulin_type
                    })
                    row_id += 1
    
    # Создаем DataFrame
    df = pd.DataFrame(input_data)
    df = df.sort_values('datetime')
    df = df.reset_index(drop=True)
    
    # Сохраняем файл
    output_path = 'assets/data/diabetes_t1_input_sparse_4days_insulin.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Разряженный 4-дневный датасет с инсулином сохранен: {output_path}")
    print(f"📊 Общее количество записей: {len(df)}")
    
    # Анализ структуры
    base_count = len(df[df['key_group'] == 'base'])
    diag_count = len(df[df['key_group'] == 'diag'])
    glu_count = len(df[df['key_group'] == 'glu'])
    insulin_count = len(df[df['key_group'] == 'insulin'])
    
    print(f"👤 Базовые признаки: {base_count}")
    print(f"🏥 Диагнозы: {diag_count}")
    print(f"🩸 Измерения глюкозы: {glu_count}")
    print(f"💉 Записи инсулина: {insulin_count}")
    
    # Анализ по дням
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    daily_stats = df.groupby('date').agg({
        'key_group': 'count'
    })
    
    print(f"\n📋 Записи по дням:")
    for date, count in daily_stats.iterrows():
        print(f"  {date}: {count['key_group']} записей")
    
    # Показываем структуру
    print(f"\n📋 Структура разряженного файла:")
    print("Первые записи (индивидуальные данные):")
    base_first = df[df['key_group'] == 'base'].head()
    for _, row in base_first.iterrows():
        print(f"  {row['datetime']} | {row['key_group']} | {row['key']} | {row['value']}")
    
    print("\nПервые записи временного ряда:")
    time_first = df[df['key_group'].isin(['glu', 'insulin'])].head(15)
    for _, row in time_first.iterrows():
        print(f"  {row['datetime']} | {row['key_group']} | {row['key']} | {row['value']}")
    
    return output_path

def load_patient_data():
    """Загружает данные пациента"""
    print("=== ЗАГРУЗКА ДАННЫХ ПАЦИЕНТА ===")
    
    # Загружаем данные веса
    weight_file = 'assets/data/D_1_type/fitbit_data/2022_04_25_all_time_export/Personal & Account/weight-2021-11-28.json'
    height_file = 'assets/data/D_1_type/fitbit_data/2022_04_25_all_time_export/Personal & Account/height-2021-11-28.json'
    
    patient_data = {}
    
    if os.path.exists(weight_file):
        with open(weight_file, 'r') as f:
            weight_data = json.load(f)
            if weight_data:
                weight_lbs = weight_data[0]['weight']  # Вес в фунтах
                weight_kg = weight_lbs * 0.453592  # Конвертация в кг
                patient_data['weight_lbs'] = weight_lbs
                patient_data['weight_kg'] = weight_kg
                patient_data['bmi_from_fitbit'] = weight_data[0]['bmi']
                patient_data['weight_date'] = weight_data[0]['date']
                print(f"Вес из Fitbit: {weight_lbs} фунтов = {weight_kg:.1f} кг (дата: {patient_data['weight_date']})")
                print(f"BMI из Fitbit: {patient_data['bmi_from_fitbit']}")
    
    if os.path.exists(height_file):
        with open(height_file, 'r') as f:
            height_data = json.load(f)
            if height_data:
                height_mm = int(height_data[0]['value'])
                patient_data['height_cm'] = height_mm / 10
                patient_data['height_date'] = height_data[0]['dateTime']
                print(f"Рост из Fitbit: {patient_data['height_cm']} см (дата: {patient_data['height_date']})")
    
    return patient_data

if __name__ == "__main__":
    dataset_file = create_sparse_4day_with_insulin()
    print(f"\n🎯 Готово! Разряженный 4-дневный датасет с инсулином: {dataset_file}") 