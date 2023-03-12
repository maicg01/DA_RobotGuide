import pandas as pd

classes_data = pd.read_csv('DSDB.csv')
# classes_data = classes_data.rename(columns={'Họ và tên': 'Name'})
# classes_data = classes_data.rename(columns={'Unnamed: 2': 'DOI'})
# classes_data.to_csv('DSDB.csv', index=False)

classes_data = classes_data.assign(Checkin=None)
classes_data.to_csv('DSDB.csv', index=False)
name_ID = classes_data.loc[classes_data['STT'] == 151]
print(name_ID)
print(classes_data.head())
value = name_ID.iloc[0,2]
print("value: ", value)

name_ID.iloc[0,3] = "OK"
print(classes_data.head())
print(name_ID)
