import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from whiteboxml.linear_models.linear_regression import LinearRegression

def test_gradient_descent_visualization():
    """
    Verifica la convergencia del modelo y genera un gráfico informativo
    utilizando el dataset de Diabetes.
    """
    # 1. Cargar el dataset de Diabetes
    diabetes = load_diabetes()
    
    # Seleccionamos la columna 2: Body Mass Index (BMI)
    # Los datos ya están normalizados por sklearn
    X = diabetes.data[:, 2:3] 
    y = diabetes.target

    # 2. Configurar y entrenar tu modelo
    # Usamos iterations en lugar de max_iter para cumplir con la cátedra
    model = LinearRegression(
        method="gd", 
        learning_rate=0.1, 
        iterations=1000, 
        normalize=True
    )
    model.fit(X, y)

    # 3. Realizar predicciones para la recta
    y_pred = model.predict(X)

    # 4. Configuración profesional del gráfico
    plt.figure(figsize=(10, 6))
    
    # Puntos de datos (Diabetes real)
    plt.scatter(X, y, color='royalblue', alpha=0.6, label='Pacientes (Datos Reales)')
    
    # Línea de tendencia (Tu modelo)
    plt.plot(X, y_pred, color='red', linewidth=3, label='Modelo de Regresión Lineal')

    # Etiquetas de ejes y títulos basados en el dataset
    plt.xlabel("Índice de Masa Corporal (BMI) Normalizado")
    plt.ylabel("Progresión de la enfermedad (1 año)")
    plt.title("Análisis de Progresión de Diabetes vs IMC - WhiteBoxML")
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Mostrar el gráfico
    plt.show()

    # Verificación técnica para el test
    assert model._trained is True
    assert len(model.theta) == 2