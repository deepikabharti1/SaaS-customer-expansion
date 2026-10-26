import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

#load saas customer dataset
accounts=pd.read_csv("ravenstack_accounts.csv")
subscriptions=pd.read_csv("ravenstack_subscriptions.csv")
feature_usage=pd.read_csv("ravenstack_feature_usage.csv")
support_tickets=pd.read_csv("ravenstack_support_tickets.csv")
churn_events=pd.read_csv("ravenstack_churn_events.csv")

#inspection of datasets
print(accounts.head(5))
print(accounts.shape)
print(accounts.columns)
print(accounts.info())

print(subscriptions.head())
print(subscriptions.shape)
print(subscriptions.info())

print(feature_usage.head())
print(feature_usage.shape)
print(feature_usage.info())

print(support_tickets.head())
print(support_tickets.shape)
print(support_tickets.info())

print(churn_events.head())
print(churn_events.shape)
print(churn_events.info())

#fix datetime
accounts['signup_date']=pd.to_datetime(accounts['signup_date'])
feature_usage['usage_date']=pd.to_datetime(feature_usage['usage_date'])
support_tickets['submitted_at']=pd.to_datetime(support_tickets['submitted_at'])
support_tickets['closed_at']=pd.to_datetime(support_tickets['closed_at'])
churn_events['churn_date']=pd.to_datetime(churn_events['churn_date'])

#check important columns datatype
print(accounts['signup_date'].dtype)
print(feature_usage['usage_date'].dtype)
print(support_tickets['submitted_at'].dtype)
print(support_tickets['closed_at'].dtype)
print(churn_events['churn_date'].dtype)

#check duplicate value in the table and fix it
print("accounts duplicates:", accounts.duplicated().sum())
print("subscriptions duplicates:" ,subscriptions.duplicated().sum())
print("feature_usage duplicates:", feature_usage.duplicated().sum())
print("support_tickets duplicates:", support_tickets.duplicated().sum())
print("churn_events duplicates:", churn_events.duplicated().sum())

#check null values in the table and fix it
print('accounts:')
print(accounts.isnull().sum())
print('\nsubscriptions:')
print(subscriptions.isnull().sum())
print('\nfeature_usage:')
print(feature_usage.isnull().sum())
print('\nsupport_tickets:')
print(support_tickets.isnull().sum())
print('\nchurn_events:')
print(churn_events.isnull().sum())

#check column names to identify target columns
print('accounts:',accounts.columns.tolist())
print('subscriptions:',subscriptions.columns.tolist())
print('feature_usage:',feature_usage.columns.tolist())
print('support_tickets:',support_tickets.columns.tolist())
print('churn_events:', churn_events.columns.tolist())

#find unique acc and subs id in table to find out the actual unique number of customers
print('accounts:', accounts['account_id'].nunique(), accounts.shape[0])
print('subscriptions:' , subscriptions['subscription_id'].nunique(),subscriptions.shape[0])

#make sure the no. of unique subscription id is same in both subscription and feature table
print('unique subscription id in subscriptions:', subscriptions['subscription_id'].nunique())
print('unique subscription id in feature usage:', feature_usage['subscription_id'].nunique())

print('feature usage rows:', feature_usage.shape[0])
print(feature_usage.columns.tolist())

#aggregate customer activity and combine dataset at account level
usage_summary=feature_usage.groupby('subscription_id').agg(
    total_usage=('usage_count', 'sum'),
    avg_usage=('usage_count','mean'),
    total_duration=('usage_duration_secs', 'sum'),
    total_error=('error_count', 'sum'),
    unique_features=('feature_name','nunique')).reset_index()
print(usage_summary.head(5))
print(usage_summary.shape)

subscription_data=subscriptions.merge(
    usage_summary,
    on='subscription_id', 
    how='left')
print(subscription_data.head(5))
print(subscription_data.shape)

usage_columns=[
    'total_usage',
    'avg_usage',
    'total_duration',
    'total_error',
    'unique_features']
#fill null with zero so that it will work
subscription_data[usage_columns]=subscription_data[usage_columns].fillna(0)

#convert subscription dates to mixed date time format
subscription_data['start_date']=pd.to_datetime(subscription_data['start_date'],format='mixed')
subscription_data['end_date']=pd.to_datetime(subscription_data['end_date'],format='mixed')

#inspect subscription data, upgarde lables, and customer acc info
print('\nsubscription values--->')
print(subscriptions['upgrade_flag'].value_counts(dropna=False))
print(subscriptions[['account_id','start_date','end_date','upgrade_flag']].head(3))
print('\naccounts inspection--->')
print(accounts.head(3))
print('\nsupport tickets--->')
print(support_tickets.head(3))
print('\nchurn events inspection--->')
print(churn_events['preceding_upgrade_flag'].value_counts(dropna=False))
print('\nplan tier-->')
print(subscriptions.groupby(['plan_tier', 'upgrade_flag'])['account_id'].count())
print('\nunique accounts -->')
print(subscriptions["account_id"].value_counts().head(10))

#validate aggregated usage data and account subscription records
print('\nusage summary-->')
print(usage_summary.shape)
print('\nsubscription data shape-->')
print(subscription_data.shape)
print('\nsubscription data unique-->')
print(subscription_data['account_id'].nunique())
print('\nsubscription varies by account_id--->')
print(subscriptions.groupby('account_id')['upgrade_flag'].nunique().value_counts())
print('\ndate range for upgrade flag--->')
print(subscriptions[['start_date','end_date']].describe())
print('\ndate range')
print(subscriptions[['start_date', 'end_date']].agg(['min','max']))

#prepare features and target variable for ML
model_data=subscription_data.copy()
model_data['upgrade_flag']=model_data['upgrade_flag'].astype(int)

x=model_data[['plan_tier','seats','mrr_amount','arr_amount','is_trial',
              'billing_frequency','auto_renew_flag','total_usage',
              'avg_usage','total_duration','total_error','unique_features']]

y=model_data['upgrade_flag']

x=pd.get_dummies(x,columns=['plan_tier','billing_frequency'],dtype=int)
#split the data
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

#train xgboost classifiaction model
model=XGBClassifier(random_state=42,eval_metric='logloss') 
model.fit(x_train,y_train)

#predict customer upgrade outcomes(x test data)
y_pred=model.predict(x_test)

#evaluate model performance using classification matrics
print('accuracy:', accuracy_score(y_test,y_pred))
print('precision:', precision_score(y_test,y_pred,zero_division=0))
print('recall:',recall_score(y_test,y_pred,zero_division=0))
print('f1:', f1_score(y_test,y_pred,zero_division=0))

###handle class imbalence bcz we need to improve the upgrade detection
model=XGBClassifier(scale_pos_weight=8.46,random_state=42,
                    eval_metric='logloss')#scale-pos-weight gives more weight to upgrade customers
model.fit(x_train,y_train)

#predict customer upgrade outcomes(x test data)
y_pred=model.predict(x_test)

print('\nytest value count-->')
print(y_test.value_counts())
print('\nhow much non upgrade customer are than upgrade customer in our training data')
scale=(y_train==0).sum()/(y_train==1).sum()#will use this value in our model so it'll pay more attention to upgrade
print(scale)#this value help the model learn the minority class(bcz upgrade is 10%only)
print('\nypred value count-->')
print(pd.Series(y_pred).value_counts())
##analyze correct and incorrect predictions
print(confusion_matrix(y_test,y_pred))

## lets see the outcome in charts to see whether upgrades are commom or rare
#countplot
sns.countplot(data=model_data, x='upgrade_flag')
plt.title('Customer Upgrade Distribution')
plt.xlabel('Upgrade Flag (0 = No, 1 = Yes)')
plt.ylabel('Number of Accounts')
plt.savefig("avg_usage_by_upgrade.png", dpi=150, bbox_inches="tight")
plt.show()

#boxplot
sns.boxplot(data=model_data, x='upgrade_flag', y='avg_usage')
plt.title('Average Usage by Upgrade Status')
plt.xlabel('Upgrade Flag (0 = No, 1 = Yes)')
plt.ylabel('Average Usage')
plt.show()

## insights: Most accounts didn't upgrade,Only a smaller group upgraded.
