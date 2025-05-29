import pandas as pd
from imblearn.over_sampling import SMOTE
# df = pd.read_csv('final2.csv')
# X = df.drop(columns=['Plant_Message_Type'])
# y = df['Plant_Message_Type']
# from collections import Counter
# print(Counter(y))
# smote = SMOTE(sampling_strategy={0:1000,1:900,2:750,3:700},random_state=42)
# X_smote, y_smote = smote.fit_resample(X, y)
# df_smote = pd.DataFrame(X_smote, columns=X.columns)
# df_smote['Plant_Message_Type'] = y_smote
# print(Counter(y_smote))
# df_smote.to_csv('final_with_smote.csv', index=False)
# dataa= pd.read_csv('final_with_smote.csv')
# dfa= pd.DataFrame(dataa)
# print(dfa.shape)




