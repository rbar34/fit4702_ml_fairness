import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from aequitas.group import Group

df = pd.read_csv("compas.csv")
test = (df[-200:]).copy()

# pre-processing
encoder = OneHotEncoder(handle_unknown='ignore')
new_x = encoder.fit_transform(df.iloc[:, :-1])

# Random forest
rf = RandomForestClassifier(n_estimators=10)
rf = rf.fit(new_x[:-200], df['two_year_recid'][:-200])

# predictions
rf_predict = rf.predict(new_x[-200:])
scores = pd.Series(rf_predict)

# rename
test.reset_index(inplace=True)
test['score'] = scores
test.rename(columns={'two_year_recid': 'label_value'}, inplace=True)

test_aq = test.loc[:, ["race", "label_value", "score"]]

g = Group()
xtab, _ = g.get_crosstabs(test_aq)
absolute_metrics = g.list_absolute_metrics(xtab)

# print(xtab[[col for col in xtab.columns if col not in absolute_metrics]])
print(xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2))
