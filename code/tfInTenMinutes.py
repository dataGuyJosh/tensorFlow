# ETL
import pandas as pd
from sklearn.model_selection import train_test_split

# Modelling
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense                   # fully connected NN layer
from sklearn.metrics import accuracy_score

# import data & setup data frames
df = pd.read_csv('data/churn.csv')

X = pd.get_dummies(df.drop(['Customer ID', 'Churn'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=2)

# print(X_trn.head(), y_trn.head(), sep='\n')

'''
Setup NN layers
- input_dim sets the number of nodes on the input layer, usually set to the number of features
- relu converts output to 0 when input is negative
''' 
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim = len(X_trn.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

# Fit & Predict
model.fit(X_trn,y_trn,epochs=100, batch_size=32)
y_pred = model.predict(X_tst)
# prediction output is a probability distribution i.e. floats between 0 & 1
y_pred = [0 if i < 0.5 else 1 for i in y_pred]

print(f'Model Accuracy: {accuracy_score(y_tst,y_pred)}')

# save model
model.save('data/tfModel')
# delete model
del model
# load model
model = load_model('data/tfModel')