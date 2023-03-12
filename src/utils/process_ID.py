import pandas as pd


def process_ID(directory, classes_data):
    int_directory = int(directory)
    name_ID = classes_data.loc[classes_data['STT'] == int_directory]
    name_ID.iloc[0,3] = "Mai"
    # classes_data.to_csv('DSDB.csv', index=False) #chay thuc te phai luu file vao
    print(name_ID)
    ROBOT_talk = "Xin chào Đồng chí " + str(name_ID.iloc[0,1]) + " " + str(name_ID.iloc[0,2])
    return ROBOT_talk
    
