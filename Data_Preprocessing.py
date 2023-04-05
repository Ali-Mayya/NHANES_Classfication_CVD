import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import re
import seaborn as sns


lab=pd.read_csv(".Prepared_NHANES/labs.csv")
diagnos=pd.read_csv(".Prepared_NHANES/Diangos_CVD.csv")

diagnos=diagnos[["SEQN","CVD"]]
df=lab.merge(diagnos,how="inner", on="SEQN")

# define regular expression pattern
pattern = r'\([^/]+/[^/]+\)$'
# get list of columns that match the pattern
cols_to_keep = [col for col in df.columns if re.search(pattern, col)]
print((cols_to_keep))
cols_to_keep.append("SEQN")
df=df[cols_to_keep]
df=df.merge(diagnos,how="inner", on="SEQN")
df=df.drop(["SEQN"],axis=1)

df1=df.dropna()
impute_mode=SimpleImputer(strategy='most_frequent')
data=pd.DataFrame(impute_mode.fit_transform(df),columns=df.columns)

def plot_heatmat_nan():
############# Heatmap for Nan values in Dataframe ###########################3
    fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,3))
    plt.subplot(1,3,1)
    sns.heatmap(df.T.isnull())
    plt.xlabel("Dataset shape ({},{})".format(*df.shape))
    plt.title("Data Before cleaning Nan ")

    plt.subplot(1,3,2)
    sns.heatmap(df1.T.isnull())
    plt.title("Data using func Dropna")
    plt.xlabel("Dataset shape ({},{})".format(*df1.shape))


    plt.subplot(1,3,3)
    sns.heatmap(data.T.isnull())
    plt.title("Using imputer ")
    plt.xlabel("Dataset shape ({},{})".format(*data.shape))

    plt.show()
    fig.tight_layout(pad=-1)

def plot_max_mean_42classes():
    #vusilize max and mean values of data

    colors = ['#446BAD','#A2A2A2']
    CVD=data[data['CVD']==1]
    non_CVD=data[data['CVD']==0]
    numbers = list(map(lambda x: x+1, range(20)))

    CVD=CVD.iloc[:,numbers]
    non_CVD=non_CVD.iloc[:,numbers]
    CVD=CVD.describe().T
    non_CVD=non_CVD.describe().T

    fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (7,7))
    plt.subplot(1,2,1)
    sns.heatmap(CVD[['max','mean']],annot = True,cmap = 'RdYlGn',linewidths = 0.7,linecolor = 'black',cbar = False,fmt = '.2f')
    plt.title('ill-CardioVescular Disorder');
    plt.subplot(1,2,2)
    sns.heatmap(non_CVD[['max','mean']],annot=True,cmap=colors, linewidths=0.7,cbar = False,fmt = '.2f')
    plt.title("healthy-Non-CVD")
    fig.tight_layout(pad=-1)
    plt.show()

def plot_Hist_Distribution_CVD(data):
    ######################Divid datset for Categorical and Numerical###################
    #####Categorical refers to columns that have distinc numbers  more than 2
    col = list(data.columns)
    categorical_features = []
    numerical_features = []
    for i in col:
        if len(data[i].unique()) > 2:
            numerical_features.append(i)
        else:
            categorical_features.append(i)

    print('Categorical Features :',*categorical_features)
    print('Numerical Features :',*numerical_features)

    plt.figure(figsize = (7,7))
    sns.histplot(data[categorical_features[0]],kde_kws = {'bw' : 1});
    title = 'Distribution : ' + categorical_features[0]
    plt.title(title)
    plt.show()
    return numerical_features

def subplot_distrubtion(numerical_features,num,data):

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15, 25))
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        sns.distplot(data[numerical_features[i]])
        title = 'Distribution : ' + numerical_features[i]
        plt.title(title)
        plt.tight_layout()
    plt.show()

# subplot_distrubtion(plot_Hist_Distribution_CVD(data),8,data)

X=(data.loc[:,data.columns !="CVD"])
y=(data.CVD)


print(y.value_counts())









#