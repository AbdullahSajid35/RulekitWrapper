import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from RulekitWrapper import RulekitWrapper

current_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(current_dir,'Datasets/Titanic-Dataset.csv'))
data = data[["Pclass", "Survived", "Sex", "Age", "SibSp", "Parch", "Fare"]]
data.reset_index(inplace=True, drop=True)
data.columns = [i.lower() for i in data.columns]
data.age = data.age.fillna(data.age.median())
data.fare = data.age.fillna(data.fare.median())
data['pclass'] = data['pclass'].astype(int)
scaler = MinMaxScaler()
data_dummies = pd.get_dummies(data.drop(["survived"], axis=1),columns=["pclass", "sex"])
data_dummies = data_dummies.drop(["sex_male"], axis=1)
data_scaled = pd.DataFrame(scaler.fit_transform(data_dummies),index=data_dummies.index,columns=data_dummies.columns)
X = data_scaled
y = data.survived.astype(int)
categorical_cols = ["pclass_1", "pclass_2", "pclass_3","sex_female"]  # your categorical columns
for col in categorical_cols:
    X[col] = X[col].astype("category")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, )

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=100, batch_size = 24, validation_split=0.2, callbacks=[early_stopping_callback])

train_acc = np.round(accuracy_score(y_train, model.predict(X_train)>0.5),3)
train_bacc = np.round(balanced_accuracy_score(y_train, model.predict(X_train)>0.5),3)
print(f"NN model train accuracy: {train_acc}")
print(f"NN model train bacc: {train_bacc}")
test_acc = np.round(accuracy_score(y_test, model.predict(X_test)>0.5),3)
test_bacc = np.round(balanced_accuracy_score(y_test, model.predict(X_test)>0.5),3)
print(f"NN model test accuracy: {test_acc}")
print(f"NN model test bacc: {test_bacc}")
clf_train_pred = (model.predict(X_train) > 0.5).astype(int).ravel()
clf_test_pred  = (model.predict(X_test)  > 0.5).astype(int).ravel()    # Test Predictions from classifier

rk = RulekitWrapper(max_growing=2, max_rule_count=3)     # Hyper-parameters to control rules generation
rk.fit(X_train, pd.Series(clf_train_pred,name='label'))

rules_list = rk.get_rules()
print(f'Total Rules: {len(rules_list)}')
for rule, w in rules_list:
  print(f'Rule: {rule},  Weight: {w}')


rules_stats = rk.rules_statistics(X_train, y_train)
print(rules_stats)

y_test_pred_rulekit = rk.predict(X_test, rules_list)
print(f'Fidelity Score: {accuracy_score(clf_test_pred, y_test_pred_rulekit)}')

clf_accuracy = accuracy_score(y_test, clf_test_pred)
rule_model_accuracy = accuracy_score(y_test, y_test_pred_rulekit)
drop_in_accuracy = (clf_accuracy - rule_model_accuracy) * 100

print(f"Classifier Accuracy: {clf_accuracy}")
print(f"Rule Model Accuracy: {rule_model_accuracy}")
print(f"Drop in Accuracy: {drop_in_accuracy}%")


sample = X_test.iloc[0]
explanation = rk.local_explainability(sample)
print(f"Single Sample Explanational Details: {explanation}")