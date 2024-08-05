import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import streamlit as st

import requests

####################################################################
#Introduccion

st.markdown("<h1 style='text-align: center; color: #4a09d6;'>Fiabilidad de los clientes</h1>", unsafe_allow_html=True)
st.markdown('''<p>&nbsp;&nbsp;&nbsp; Un proveedor de telecomunicaciones ah segmentado su base clientes según patrones de uso del servicio, catagorizando a los clientrs en 4 grupos.
            
            CustomerCategory
            1. Basic service
            2. E-Service    
            3. Plus service
            4. Full service

Nuestro <strong>objetivo es construir un clasificador para predecir el servicio para los casos desconocidos</strong>. <br> Usare un tipo específico de clasificación llamado K-vecino más cercano (k-nearest neighbour).</p>''', unsafe_allow_html=True)
st.markdown("--- \n ---", unsafe_allow_html=True)
####################################################################
#Vista general de datos

st.markdown("<h3>Datos generales<h3>", unsafe_allow_html=True)

df = pd.read_csv("teleCust100t.csv")

st.dataframe(df.describe())

####################################################################
#Preparando los datos
service_num = df['custcat'].replace({1: 'Basic service', 2: 'E-service', 3: 'Plus service', 4: 'Full service'})
service_num2 = dict(service_num.value_counts())
for i in service_num2:
    st.write('En', f'**-{i}-**', 'hay', service_num2[i], 'clientes')
st.markdown('<br><br>', unsafe_allow_html=True)

#Normalizacion
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values.astype(float)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = df['custcat'].values

#Separado entre entrenamiento y testeo
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=4)

####################################################################
#Modelo de clasificacion KNN
if st.button('Ver el desarrollo del modelo'):
    pass
from sklearn.neighbors import KNeighborsClassifier
k = st.number_input('k = ? ', min_value=1, max_value=100, value=4)#k vecinos mas cercanos

#Entrenamiento
neigh =  KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train) 

#Prediccion
yhat = neigh.predict(X_test)

#Evaluacion de presicion
from sklearn import metrics
#Predicciones correctas/total de predicciones
st.markdown("<font style='font-size: 20px;'>Predicciones correctas / Predicciones totales</font> <br> Si la presicion del entrenamiento aumenta y la de presicion de testeo disminuye, significa que el modelo a memorizado. Eso es malo", unsafe_allow_html=True)
st.markdown(f"Que tan bien el modelo ha aprendido <br> <font style='font-size: 20px;'><strong>Presicion del entrenamiento: <font style='color: green;'>{metrics.accuracy_score(y_train, neigh.predict(X_train))}</font></strong></font><br>", unsafe_allow_html=True)
st.markdown(f"Que tan bien el modelo genera nuevos datos <br> <font style='font-size: 20px;'><strong>Presicion del testeo: <font style='color: green;'>{metrics.accuracy_score(y_test, yhat)}</font></strong></font>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

#Differents k
ks = 12
mean_acc = np.zeros((ks-1))
std_acc = np.zeros((ks-1))
st.write('Promedio de presicion:')
for n in range(1, ks):
    #Train model and predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test,  yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    st.write(f'k = {n} -->' ,mean_acc[n-1])

mean_acc_str = f', '.join([f'{x: .3f}' for x in mean_acc])
st.write(mean_acc_str)

#Graficamos la presicion del modelo con sus diferentes k
fig, ax = plt.subplots() #figura con ejes
ax.plot(range(1, ks), mean_acc, 'g')
ax.fill_between(range(1, ks), mean_acc - 1*std_acc, mean_acc + 1*std_acc, alpha=0.1)
ax.fill_between(range(1, ks), mean_acc - 3*std_acc, mean_acc + 3*std_acc, alpha=0.1, color='green')
ax.legend(('accuraacy', '+/-  1xstd', '+/- 3xstd'))
ax.set_ylabel('Accuracy')
ax.set_xlabel('Number of Neighbors (K)')
plt.tight_layout()
st.pyplot(fig)

st.markdown(f"La mejor presicion fue {mean_acc.max()} con k = {np.argmax(mean_acc) + 1}", unsafe_allow_html=True)

####################################################################
#Guardado modelo entrenado
import pickle

#Guardado del modelo entrenado
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(neigh, f)
#Carga modelo entrenado
with open('knn_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

#Prediccion con el modelo cargado
yhat_loaded = loaded_model.predict(X_test)
loaded_model_accuracy = metrics.accuracy_score(y_test,  yhat_loaded)

st.markdown(f"Presicion del modelo: <font style='color: green;'>{loaded_model_accuracy}</font>", unsafe_allow_html=True)

####################################################################
#Prediccion de nuevos datos ingresados

#Formulario nuevos datos ingresados
region = st.number_input('Region', min_value=0, max_value=10, value=1)
tenure = st.number_input('Tenure', min_value=0, max_value=100, value=1)
age = st.number_input('Age', min_value=0, max_value=100, value=18)
marital = st.number_input('Marital Status', min_value=0, max_value=1, value=0)
address = st.number_input('Address', min_value=0, max_value=50, value=1)
income = st.number_input('Income', min_value=0, max_value=200, value=50)
ed = st.number_input('Education Level', min_value=0, max_value=10, value=1)
employ = st.number_input('Employment', min_value=0, max_value=50, value=1)
retire = st.number_input('Retired', min_value=0, max_value=1, value=0)
gender = st.number_input('Gender', min_value=0, max_value=1, value=0)
reside = st.number_input('Residency', min_value=0, max_value=10, value=1)

#Prooceso de los datos ingresados
user_data = np.array([[region, tenure, age, marital, address, income, ed, employ, retire, gender, reside]])
user_data = preprocessing.StandardScaler().fit(X).transform(user_data)

#Prediccion
prediction = loaded_model.predict(user_data)
st.markdown(f"<font style='font-size: 50px;'>Categoría de servicio predicha: <font  style='color: blue;'>{prediction[0]}</font></font>", unsafe_allow_html=True)