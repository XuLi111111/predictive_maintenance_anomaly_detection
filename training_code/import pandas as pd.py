import pandas as pd

df = pd.read_csv('sensor.csv')
df.drop(columns=['sensor_15', 'sensor_50', 'sensor_51'], inplace=True)
df.ffill(inplace=True)

label_map = {
    'NORMAL': 0,
    'RECOVERING': 1,
    'BROKEN': 2
}

df['label'] = df['machine_status'].map(label_map)

df.to_csv('sensor_processed.csv', index=False)
print("Done")