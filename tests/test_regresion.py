import numpy as np
import matplotlib.pyplot as plt
from whiteboxml.linear_models import LinearRegression

# Generar datos de prueba: una línea con ruido
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100)

# Entrenar el modelo
model = LinearRegression(learning_rate=0.1, iterations=500)
model.fit(X, y)

# Predecir valores para graficar la recta
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# Mostrar resultados
plt.scatter(X, y, color='blue', label='Datos')
plt.plot(X_new, y_predict, color='red', label='Regresión Lineal')
plt.legend()
plt.title("Prueba de Regresión Lineal - WhiteBoxML")
plt.show()

print(f"Pesos calculados: {model.theta}")