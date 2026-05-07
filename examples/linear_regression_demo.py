# -*- coding: utf-8 -*-
"""
Script de demostración para el modelo de Regresión Lineal.
Muestra el uso de la solución analítica y el descenso por gradiente
utilizando datasets reales.

:authors: Equipo Regresión Lineal
:date: mayo 2026
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from whiteboxml.linear_models.linear_regression import LinearRegression

def run_demo():
    """
    Ejecuta la comparación de métodos de optimización y visualiza resultados.
    """
    print("--- WhiteBoxML: Demo de Regresión Lineal ---")
    
    # 1. Carga de datos (California Housing - Predicción de precios de casas)
    print("\nCargando dataset de California Housing...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    # 2. Configuración y entrenamiento de modelos
    print("Entrenando modelos...")
    
    # Modelo con Solución Analítica
    model_an = LinearRegression(method="analytic")
    model_an.fit(X, y)
    print("✓ Solución analítica completada.")

    # Modelo con Descenso por Gradiente Batch (Normalizado)
    # Se cambió max_iter por iterations
    model_gd_batch = LinearRegression(
        method="gd", 
        mode="batch", 
        learning_rate=0.01, 
        iterations=1000, 
        normalize=True
    )
    model_gd_batch.fit(X, y)
    print(f"✓ GD Batch completado en {len(model_gd_batch.cost_history)} iteraciones.")

    # Modelo con Descenso por Gradiente Estocástico (SGD)
    # Se cambió max_iter por iterations
    model_sgd = LinearRegression(
        method="gd", 
        mode="sgd", 
        learning_rate=0.001, 
        iterations=2000, 
        normalize=True
    )
    model_sgd.fit(X, y)
    print(f"✓ SGD completado.")

    # 3. Visualización de la convergencia (Historial de Costo)
    print("\nGenerando gráficos de convergencia...")
    plt.figure(figsize=(12, 6))
    
    # Graficamos la curva de aprendizaje
    plt.plot(model_gd_batch.cost_history, label="Batch GD (lr=0.01)", linewidth=2)
    plt.plot(model_sgd.cost_history, label="SGD (lr=0.001)", alpha=0.6, linestyle='--')
    
    plt.title("Curva de Aprendizaje: Evolución del Error Cuadrático Medio", fontsize=14)
    plt.xlabel("Número de Iteraciones", fontsize=12)
    plt.ylabel("Costo (MSE) - Escala Log", fontsize=12)
    plt.yscale('log')  # Ayuda a ver la caída drástica al principio
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    print("\nPRO TIP: El gráfico muestra cómo el error disminuye a medida que el modelo 'aprende'.")
    print("Mostrando gráfico. Cierra la ventana para terminar.")
    plt.show()

if __name__ == "__main__":
    run_demo()