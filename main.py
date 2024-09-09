import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense

# 1. Lectura de los Datos
data = pd.read_csv('altura_peso.csv')
x = data['Altura'].values
y = data['Peso'].values

# 2. Preprocesamiento de Datos
x = x.reshape(-1, 1)  # Reshape para que sea compatible con Keras

# 3. Implementación del Modelo
model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento del Modelo
model.fit(x, y, epochs=100)

# 4. Visualización de Resultados
plt.scatter(x, y, color='blue', label='Datos Reales')
plt.plot(x, model.predict(x), color='red', label='Predicción')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title('Regresión Lineal con Keras')
plt.legend()
plt.show()

# 5. Predicción
altura_nueva = np.array([[170]])
peso_predicho = model.predict(altura_nueva)
print(f'El peso predicho para una altura de {altura_nueva[0][0]} cm es {peso_predicho[0][0]:.2f} kg')