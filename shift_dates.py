import pandas as pd
import glob
from datetime import datetime

def shift_dataset_dates():
    csv_files = glob.glob('data/*.csv')
    
    # We want to shift the dates so the last date is yesterday
    # Find the maximum date across cleaned datasets to determine the shift
    max_date = None
    for f in ['data/cleaned_hourly.csv', 'data/cleaned_daily.csv']:
        try:
            df = pd.read_csv(f, usecols=['Datetime'])
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            m = df['Datetime'].max()
            if max_date is None or m > max_date:
                max_date = m
        except Exception:
            pass
            
    if max_date is None:
        print("No max date found.")
        return

    today = pd.to_datetime(datetime.now().date())
    yesterday = today - pd.Timedelta(days=1)
    
    delta_days = (yesterday - max_date).days
    # Shift by exact number of weeks to preserve day of week
    delta_weeks = delta_days // 7
    shift_days = delta_weeks * 7
    
    shift = pd.Timedelta(days=shift_days)
    print(f"Shifting dates by {shift_days} days to align with yesterday ({yesterday.date()})")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime']) + shift
                df.to_csv(file, index=False)
                print(f"Shifted {file}")
        except Exception as e:
            print(f"Failed to shift {file}: {e}")

if __name__ == '__main__':
    shift_dataset_dates()
