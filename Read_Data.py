import os
import pandas
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import zipfile
# with zipfile.ZipFile("NHANES dataset rar.zip","r") as zipref:
#     zipref.extractall("NHANES_DataSet")

data_path= "./NHANES_DataSet"
NHANES_sets_dict={0:"demographic.csv",
                  1:"diet.csv",
                  2:"examination.csv",
                  3:"labs.csv",
                  4:"medications.csv",
                  5:"questionnaire.csv"}
keys=(2,3,5)
NHANES_diabets_CVD=NHANES_sets_dict={
                  2:"examination.csv",
                  3:"labs.csv",
                  5:"questionnaire.csv"}
print(NHANES_diabets_CVD)
dict_path=[]
for files in os.listdir(data_path):

    if files in NHANES_diabets_CVD.values():
        dict_path.append(os.path.join(data_path,files))
        print(files)

Dataframes=list()
for path in dict_path:
    Dataframes.append(pd.read_csv(path))

# print(Dataframes[1])


def prepare_data(df,dic,name):
    df=df.select_dtypes(['number'])
    df=df.dropna(thresh=0.5*len(df),axis=1)
    df=df.drop_duplicates()
    listofcol=df.columns.values
    print(listofcol)
    df=df.dropna(thresh=0.7*len(listofcol))
    c=0
    new_columns=[]
    drop_list=[]
    dic=dic.drop_duplicates()
    for item in listofcol:
        if item  in dic["Variable Name"].values:
            new_columns.append(dic.loc[dic["Variable Name"]==item]["Variable Description"].values[0])
            c=c+1
        else:
            drop_list.append(item)
    new_columns[0] = "SEQN"
    df=df.drop(drop_list,axis=1)

    # print(drop_list)
    print(new_columns)
    df.columns = new_columns
    print(c, len(df.columns))

    folder_data=".Prepared_NHANES"

    if not os.path.exists(folder_data):
        os.makedirs(folder_data)
    print(os.path.join(folder_data,name))
    df.to_csv(os.path.join(folder_data,name),index=False)



NHANES_CODES_Dict=pd.read_excel("NAHNES 2014 Dictionary.xlsx")

# prepare_data(Dataframes[0],NHANES_CODES_Dict,"examination.csv")
# prepare_data(Dataframes[1],NHANES_CODES_Dict,"labs.csv")
#

df=Dataframes[2]
# print(df.columns)
dic_diangosis= {"DIQ010": "Diabetes" ,
"DIQ160" :"Prediabetes" ,
"MCQ160B" : "CVD-(Congestive Heart Failure)",
"MCQ160C" : "CVD-(Coronary Heart Disease)",
"MCQ160D" : "CVD-(Angina Pectoris)",
"MCQ160E" :"CVD-(Heart Attack)",
"MCQ160F" :" CVD-(Stroke))"
}

############################################################################
#prepare Outcome data for labeling the target set
# set y lable to 0 for healthy case and for 1 for one of CVD disease at least
Ques=df[['SEQN','MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F']]

Ques=Ques.loc[Ques["MCQ160B"].notna() &(Ques["MCQ160B"] != 9)]
Ques=Ques.loc[Ques["MCQ160D"].notna()& (Ques["MCQ160D"] != 9)]
Ques=Ques.loc[Ques["MCQ160C"].notna()& (Ques["MCQ160C"] != 9)]
Ques=Ques.loc[Ques["MCQ160E"].notna() & (Ques["MCQ160E"] != 9)]
Ques=Ques.loc[Ques["MCQ160F"].notna() & (Ques["MCQ160F"] != 9)]

print(Ques.head)
print(Ques["MCQ160F"].value_counts())

Ques=Ques.replace([2],0)
print(Ques["MCQ160B"].value_counts())
