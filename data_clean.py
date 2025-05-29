import pandas as pd
import numpy as np
data= pd.read_csv('alien_plant_communication_dataset.csv')
df= pd.DataFrame(data)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
#    remove outliers
col = ["Leaf_Vibration_Hz", "Pollen_Scent_Complexity", "Bioluminescence_Intensity_Lux", "Sunlight_Exposure_Hours"]
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
IQR = q3 - q1
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
for c in col:
    # print(f"Lower Bound of {c} = {lower_bound[c]}")
    # print(f"Upper Bound of {c} = {upper_bound[c]}")
    df = df[(df[c] >= lower_bound[c]) & (df[c] <= upper_bound[c])]
print(df.shape)
# df.to_csv('cleaned_data.csv', index=False)
    #   data scaling
data_c= pd.read_csv('cleaned_data.csv')
df_c= pd.DataFrame(data_c)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
col = ["Leaf_Vibration_Hz", "Pollen_Scent_Complexity", "Bioluminescence_Intensity_Lux", "Sunlight_Exposure_Hours","Root_Signal_Strength_mV","Growth_Rate_mm_day","Ambient_Temperature_C","Soil_Moisture_Level"]
df_c[col] = scaler.fit_transform(df_c[col])
print(df_c.describe())
# df_c.to_csv('cleand&scaled_data.csv', index=False)
#   data encoding
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df_c['Plant_Message_Type'] = le.fit_transform(df_c['Plant_Message_Type'])
plant_message_type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(plant_message_type_mapping)
df_c['Plant_Message_Type'].value_counts()
df_c.drop(columns=['Plant_ID'], inplace=True)
df_c.to_csv('final2.csv', index=False)