#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import collections
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, auc
import xgboost as xgb

#Load datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
dc = df_train

#Preprocessing by enoding categorical values and using dummies for the string values
label_encoder = preprocessing.LabelEncoder()

#Train dataset
dc['international_plan'] = label_encoder.fit_transform(dc['international_plan'])
dc['voice_mail_plan'] = label_encoder.fit_transform(dc['voice_mail_plan'])
dc['churn'] = label_encoder.fit_transform(dc['churn'])
dc=pd.concat([pd.get_dummies(dc.area_code),dc],axis=1)
dc=pd.concat([pd.get_dummies(dc.state),dc],axis=1)

#Test dataset
df_test['international_plan'] = label_encoder.fit_transform(df_test['international_plan'])
df_test['voice_mail_plan'] = label_encoder.fit_transform(df_test['voice_mail_plan'])
df_test=pd.concat([pd.get_dummies(df_test.area_code),df_test],axis=1)
df_test=pd.concat([pd.get_dummies(df_test.state),df_test],axis=1)

#We create new columns by adding oll the total_minutes, total_charge and total_calls and we tranform minutes column to hours
dc['minutes']=dc.total_day_minutes + dc.total_eve_minutes + dc.total_night_minutes + dc.total_intl_minutes
dc['calls']=dc.total_day_calls + dc.total_eve_calls + dc.total_night_calls + dc.total_intl_calls
dc['charge']=dc.total_day_charge + dc.total_eve_charge + dc.total_night_charge + dc.total_intl_charge
dc['hours'] = dc.minutes/60

df_test['minutes']=df_test.total_day_minutes + df_test.total_eve_minutes + df_test.total_night_minutes + df_test.total_intl_minutes
df_test['calls']=df_test.total_day_calls + df_test.total_eve_calls + df_test.total_night_calls + df_test.total_intl_calls
df_test['charge']=df_test.total_day_charge + df_test.total_eve_charge + df_test.total_night_charge + df_test.total_intl_charge
df_test['hours'] = df_test.minutes/60

#Drop the columns tha we processed above
dc = dc.drop(['state', 'area_code', 'minutes'],axis=1)
X_test = df_test.drop(['state', 'area_code', 'minutes', 'id'],axis=1)

#Corellation heatmap
cor = dc.drop(['churn'],axis=1)
sns.heatmap(cor.corr(), cmap = "Blues")

#Pairplot
sns.pairplot(df[['account_length', 'international_plan', 'voice_mail_plan', 'total_day_minutes', 'total_eve_minutes', 'total_eve_calls', 'total_night_minutes', 'total_night_calls', 'total_intl_minutes', 'total_intl_calls', 'number_customer_service_calls', 'churn']], hue='churn', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)

#Create three classes eina numerical columns high, medium an low and evaluate their contribution to churn

dt=dc

dt['total_day_minutes'] = pd.cut(dt['total_day_minutes'], bins=[0,150,250,400], labels=["Low", "Mid", "High"])
dt['total_eve_minutes'] = pd.cut(dt['total_eve_minutes'], bins=[0,150,250,400], labels=["Low", "Mid", "High"])
dt['total_night_minutes'] = pd.cut(dt['total_night_minutes'], bins=[0,150,250,400], labels=["Low", "Mid", "High"])

dt['total_day_calls'] = pd.cut(dt['total_day_calls'], bins=[0,50,150,200], labels=["Low", "Mid", "High"])
dt['total_eve_calls'] = pd.cut(dt['total_eve_calls'], bins=[0,50,150,200], labels=["Low", "Mid", "High"])
dt['total_night_calls'] = pd.cut(dt['total_night_calls'], bins=[0,50,150,200], labels=["Low", "Mid", "High"])


#To analyse categorical feature distribution
categorical_features = ['international_plan', 'voice_mail_plan']
ROWS, COLS = 2, 2
fig, ax = plt.subplots(ROWS, COLS,  figsize=(18, 20))
row, col = 0, 0
for i, categorical_feature in enumerate(categorical_features):
  if col == COLS - 1: row += 1
  col = i % COLS
  dc[dc.churn== 0][categorical_feature].value_counts().plot(kind='bar', width=.5, ax=ax[row, col], color='blue', alpha=0.5).set_title(categorical_feature)
  dc[dc.churn== 1][categorical_feature].value_counts().plot(kind='bar', width=.3, ax=ax[row, col], color='orange', alpha=0.7).set_title(categorical_feature)
  plt.legend(['No Churn', 'Churn'])
  fig.subplots_adjust(hspace=0.7)

categorical_features = ['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_day_calls',
                          'total_eve_calls', 'total_night_calls']
ROWS, COLS = 2, 3
fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 20))
row, col = 0, 0
for i, categorical_feature in enumerate(categorical_features):
    if col == COLS - 1: row += 1
    col = i % COLS
    dt[dt.churn == 0][categorical_feature].value_counts().plot(kind='bar', width=.5, ax=ax[row, col], color='blue',
                                                                 alpha=0.5).set_title(categorical_feature)
    dt[dt.churn == 1][categorical_feature].value_counts().plot(kind='bar', width=.3, ax=ax[row, col], color='orange',
                                                                 alpha=0.7).set_title(categorical_feature)
    plt.legend(['No Churn', 'Churn'])
    fig.subplots_adjust(hspace=0.7)

#Split the train dataset
X = dc.drop(['churn'],axis=1).values
y=dc['churn'].values

#Check the balance of the churn in dataset
dt = df_train
dt['churn'] = label_encoder.fit_transform(dt['churn'])
fig = plt.figure(figsize=(20,5))
sns.countplot(y='churn', data=dt)
print(dt.churn.value_counts())

#Apply an oversampling method to face the inbalance of the churn in our dataset
rd = RandomOverSampler(random_state=0)
X, y = rd.fit_sample(X, y)

#after Random Oversamplig:
collections.Counter(y)

#Enrich sklearn function in order to calculate Accuracy, Recall, Precision, F1_score, Specificity, Confusion matrix and ROC Curve
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    thresh = 0.5

    y_pred = cross_val_predict(estimator, X, y, cv=cv)
    conf = confusion_matrix(y, y_pred)
    # Metrics for evaluations of the results
    aucc = roc_auc_score(y, y_pred)
    recall = recall_score(y, (y_pred > thresh))
    precision = precision_score(y, (y_pred > thresh))
    F1_score = 2 * precision * recall / (precision + recall)
    specificity = sum((y_pred < thresh) & (y == 0)) / sum(y == 0)

    print('Accuracy of %s %.3f' % (title, max(test_scores_mean)))
    print('Aucc %.3f' % (aucc))
    print('Recall %.3f' % (recall))
    print('Precision %.3f' % (precision))
    print('F1_score %.3f' % (F1_score))
    print('Specificity %.3f' % (specificity))
    print(conf)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' ROC curve graph')
    plt.legend(loc="lower right")
    plt.show()

    return plt

#Compare ensemble methods with Bagging Classifier in a boxplot

# get a list of models to evaluate
def get_models():
    models = dict()
    models['RF'] = BaggingClassifier(base_estimator=RandomForestClassifier())
    models['GB'] = BaggingClassifier(base_estimator=GradientBoostingClassifier(learning_rate=0.07))
    models['Ada'] = BaggingClassifier(base_estimator=AdaBoostClassifier())
    models['ET'] = BaggingClassifier(base_estimator=ExtraTreesClassifier())
    models['DT'] = BaggingClassifier(base_estimator=DecisionTreeClassifier())

    return models


# evaluate a given model using cross-validation
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

#Apply models

#Without Bagging
g = plot_learning_curve(RandomForestClassifier(),"RandomForestClassifier",X,y,cv=10)
g = plot_learning_curve(ExtraTreesClassifier(),"ExtraTreesClassifier",X,y,cv=10)
g = plot_learning_curve(AdaBoostClassifier(),"AdaBoostClassifier",X,y,cv=10)
g = plot_learning_curve(GradientBoostingClassifier(learning_rate = 0.07),"GradientBoostingClassifier",X,y,cv=10)
g = plot_learning_curve(DecisionTreeClassifier(),"DecisionTreeClassifier",X,y,cv=10)
g = plot_learning_curve(xgb.XGBClassifier(),"XGBClassifier",X,y,cv=10)

#Bagging method
g = plot_learning_curve(BaggingClassifier(base_estimator=RandomForestClassifier()),"BaggingClassifier with RandomForestClassifier",X,y,cv=10)
g = plot_learning_curve(BaggingClassifier(base_estimator=GradientBoostingClassifier(learning_rate = 0.07)),"BaggingClassifier with GradientBoostingClassifier",X,y,cv=10)
g = plot_learning_curve(BaggingClassifier(base_estimator=AdaBoostClassifier()),"BaggingClassifier with AdaBoostClassifier",X,y,cv=10)
g = plot_learning_curve(BaggingClassifier(base_estimator=ExtraTreesClassifier()),"BaggingClassifier with ExtraTreesClassifier",X,y,cv=10)
g = plot_learning_curve(BaggingClassifier(base_estimator=DecisionTreeClassifier()),"BaggingClassifier with DecisionTreeClassifier",X,y,cv=10)
