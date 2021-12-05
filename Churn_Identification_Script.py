# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import io
import seaborn as sns


# %%
df= pd.read_csv('Datarobot Churn data.csv')


# %%
df['is_churn'] = np.where(df['is_churn'].str.contains('yes'), 1, 0)


# %%
df = df.drop(columns = 'SHOP_ID')


# %%
df_train = df
df_train.head()


# %%
df_train = df_train.fillna(0)


# %%
df_train2 = df_train
correlations = df_train2.corrwith(df_train.is_churn)
correlations = correlations[correlations!=1]
positive_correlations = correlations[correlations >0].sort_values(ascending = False)
negative_correlations =correlations[correlations<0].sort_values(ascending = True)


# %%
correlations.plot.bar(
        figsize = (50, 25), 
        fontsize = 15, 
        color = '#ec838a',
        rot = 45, grid = True)
plt.title('Correlation with Churn Rate \n',
horizontalalignment="center", fontstyle = "normal", 
fontsize = "25", fontfamily = "sans-serif")
plt.savefig('Correlation with ChurnRate_2.png')


# %%
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree


# %%
x = df_train.drop(columns='is_churn')
y = df_train['is_churn']


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# %%
clf = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth = 3)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# %%
feature_col = list(x)
print(list(x))


# %%
fig = plt.figure(figsize=(30,30))
_ = tree.plot_tree(clf, 
                   feature_names=feature_col,  
                   class_names=['0','1'],
                   filled=True)


# %%
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 10)
clf_entropy = clf_entropy.fit(x_train,y_train)
y_pred = clf_entropy.predict(x_test)


# %%
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_entropy, 
                   feature_names=feature_col,  
                   class_names=['0','1'],
                   filled=True)
plt.savefig('entropy_2.png')


# %%
for i, column in enumerate(df_train.drop('is_churn', axis =1)):
    print('Importance of feature {}: {:.4f}'.format(column,clf_entropy.feature_importances_[i]))

    fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [clf_entropy.feature_importances_[i]]})

    try:
        final_fi_6 = pd.concat([final_fi_6,fi], ignore_index=True)
    except:
        final_fi_6 = fi

final_fi_6 = final_fi_6.sort_values('Feature Importance Score', ascending= False).reset_index()
final_fi_6.to_csv('Importance_Score.csv')


# %%
from sklearn.preprocessing import KBinsDiscretizer


# %%
cols = ['active_length', 'is_churn', 'Parcel_count',
       'Average_days_between_parcels', 'ISD', 'SUBURBS', 'OSD',
       'Success_Rate_of_ISD', 'Success_Rate_of_SUBURB',
       'Success_Rate_of_OSD_HUB', 'SLA_AVG', 'Issue_parcels',
       'number_of_times_issues_raised', 'Issue_resolution_tat',
       'second_mile_tat_hr', 'E2E_TAT_day', 'E2E_TAT_ISD_day',
       'E2E_TAT_SUB_day', 'E2E_TAT_OSD_day', 'Extend_of_breach_in_hr',
       'Extend_of_breach_in_hr_ISD', 'Extend_of_breach_in_hr_SUB',
       'Extend_of_breach_in_hr_OSD', 'return_tat_in_days', 'issues_raised',
       'issues_resolved', 'area_changes', 'damaged',
       'creation_to_received_in_hours']


# %%
kbins = KBinsDiscretizer(n_bins = 2, encode = 'ordinal', strategy ='uniform')
df_train_transformed = kbins.fit_transform(df_train)


# %%
df_transformed_1 = pd.DataFrame(df_train_transformed, columns=cols)


# %%
from mlxtend.frequent_patterns import apriori, association_rules


# %%
churn_reasons = apriori(df_transformed_1, min_support = 0.1, use_colnames = True)

rules = association_rules(churn_reasons, metric ="confidence", min_threshold = 0.1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())
rule_csv = rules.to_csv('rules_original.csv')


# %%
print(rules)


# %%
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")


# %%
for index,row in rules.iterrows():
    if str(row['consequents']) == 'is_churn':
        print(row)


# %%
df_ischurn = rules[rules['consequents']=='is_churn']


# %%
df_exportcsv = df_ischurn.to_csv('Churn_Rules.csv')


