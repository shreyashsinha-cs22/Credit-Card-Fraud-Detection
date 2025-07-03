import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
import streamlit as st

credit_card_data=pd.read_csv('creditcard.csv')
legit=credit_card_data[credit_card_data.Class == 0]
fraud=credit_card_data[credit_card_data.Class == 1]

legit_sample=legit.sample(n=492)
new_dataset=pd.concat([legit_sample,fraud],axis =0)

x = new_dataset.drop(columns='Class' ,axis=1)
y=new_dataset['Class']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)
model=LogisticRegression(solver='liblinear',max_iter=1000)
model.fit(x_train, y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction, y_train)

x_test_prediction = model.predict(x_test)
testing_data_accuracy=accuracy_score(x_test_prediction, y_test)

# web app
st.title("Credit card fraud detection")
input_df=st.text_input('Enter all required features values')
input_df_splited=input_df.split(',')
submit=st.button("submit")
# abb hume batana h ke jo transaction h wo fraud h ya legit h, par usse humlogo iss inpput feature feature ko machiine understandable form me convert krna h ya usko dusre bhasa me numpy array bhi bol skte h

if submit:
    feature=np.asarray(input_df_splited, dtype=np.float64)
    prediction=model.predict(feature.reshape(1,-1))
    # abb chec krege ke prediction ke all index that is starting from 0
    if prediction[0] == 0:
        st.write("legit transaction")
    else:
        st.write("fraud transaction")