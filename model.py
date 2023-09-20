import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("compas.csv")

# pre-processing
encoder = OneHotEncoder(handle_unknown='ignore')
new_x = encoder.fit_transform(df.iloc[:, :-1])

# Random forest
rf = RandomForestClassifier(n_estimators=10)
rf = rf.fit(new_x[:-200], df['two_year_recid'][:-200])

# Neural net
nn = MLPClassifier(hidden_layer_sizes=(5, 2), max_iter=1000)
nn.fit(new_x[:-200], df['two_year_recid'][:-200])

# predictions
rf_predict = rf.predict(new_x[-200:-1])
nn_predict = nn.predict(new_x[-200:-1])

values = pd.Series(rf_predict), pd.Series(nn_predict)
results = pd.DataFrame(values, index=['rf', 'nn'])
print(results)
