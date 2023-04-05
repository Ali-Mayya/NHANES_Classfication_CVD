from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from Data_Preprocessing import X,y
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import re

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

print('X shape:', X.shape)
print('y shape:', y.shape)

###################################################################################
def confusion(y_test, y_pred):
    conf = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['True[0]', 'True[1]'], columns=['Predict[0]', 'Predict[1]'])
    print('Confusion Matrix:')
    print(conf)
    plt.figure(0, figsize=(5, 5))
    num = np.sum(confusion_matrix(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred) / num, annot=True, fmt=".2%")
    plt.title("Confusion matrix after Oversampling by SMOTE")
    plt.show()
    return conf

#####################################################################################
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=12)

#####################################################################################
def model_XGB(X_train,y_train,X_test,y_test):

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('Accuracy Score:%.2f%% ' %(model.score(X_test, y_test)*100))
    print('Prediction:', y_pred)
    print("Classification Report ", classification_report(y_test, y_pred))
    confusion(y_test, y_pred)


    # Features selected by XGBoost
    keys = list(model.get_booster().feature_names)
    values = list(model.feature_importances_)
    data = pd.DataFrame(data=values, index=keys, columns=["Importance"]).sort_values(by="Importance", ascending=False)
    # Top 24 features
    xgbfs_ = data[:24]
    # Plot feature score
    xgbfs_.sort_values(by='Importance').plot(kind='barh', figsize=(10, 8), color='#4B4E6D')
    plt.title("Top 24 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


def Over_sampling_SMORE(X_train,y_train,X_test,y_test):
##### over Sampling the imbalanced data#####################################

    smote=SMOTE()
    X_train_smt,y_train_smt=smote.fit_resample(X_train,y_train)
    X_test_smt,y_test_smt=smote.fit_resample(X_test,y_test)

    X_train_sm = pd.DataFrame(X_train_smt, columns=X.columns)
    X_test_sm = pd.DataFrame(X_test_smt, columns=X.columns)
    return X_train_sm,y_train_smt,X_test_sm,y_test_smt


X_train_sm,y_train_smt,X_test_sm,y_test_smt=Over_sampling_SMORE(X_train,y_train,X_test,y_test)
###################### Featuer Selection ##############################
# model_XGB(X_train,y_train,X_test,y_test)
model_XGB(X_train_sm,y_train_smt,X_test_sm,y_test_smt)

#######################################################################


def model_LR(X_train,y_train,X_test,y_test):

    model = LogisticRegression(max_iter=100, solver='lbfgs', class_weight='balanced', random_state=11)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Accuracy Score:%.2f%% ' % (model.score(X_test, y_test)*100))
    print('Prediction:', y_pred)
    print(classification_report(y_test, y_pred))
    confusion(y_test, y_pred)


# model_LR(X_train_sm,y_train_smt,X_test_sm,y_test_smt)
########################################################################3

def model_RFC(X_train,y_train,X_test,y_test):
    model=RandomForestClassifier(n_estimators=150,criterion='gini',max_depth=5,random_state=11)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('Accuracy Score:%.2f%% ' % (model.score(X_test, y_test)*100))
    print('Prediction:', y_pred)
    print("Classification Report ",classification_report(y_test, y_pred))
    confusion(y_test, y_pred)

    feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['Importance'])
    top_24 = feature_importances.sort_values(by='Importance', ascending=False).iloc[:24]
    top_24.sort_values(by='Importance').plot(kind="barh")
    plt.title("Top 24 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

model_RFC(X_train_sm,y_train_smt,X_test_sm,y_test_smt)

###############################################################################33






#
# #
# # >>> from sklearn.model_selection import cross_val_score
# # >>> clf = svm.SVC(kernel='linear', C=1, random_state=42)
# # >>> scores =
# # Oversampling with SMOTE¶
