# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import List, Tuple

class LinearRegression():
    """
    Implementación de Regresión Lineal Múltiple.
    Cumple con los requisitos de: solución analítica (Ecuación Normal) 
    y optimización iterativa (Descenso por Gradiente en tres variantes).
    """

    def __init__(
        self,
        method: str = "analytic",
        learning_rate: float = 1e-3,
        iterations: int = 10000,
        tol: float = 1e-6,
        normalize: bool = False,
        mode: str = "batch",
        batch_size: int = 32
    ) -> None:
        """
        Inicializa el modelo con hiperparámetros de control.
        'iterations' unifica el criterio de parada máximo solicitado por el proyecto.
        """
        self.method = method            # Define si se usa Álgebra Lineal o Optimización
        self.learning_rate = learning_rate  # Tamaño del paso en la actualización de pesos
        self.iterations = iterations    # Límite de iteraciones para evitar bucles infinitos
        self.tol = tol                  # Diferencia mínima entre pesos para asumir convergencia
        self.normalize = normalize      # Flag para estandarización de variables (Z-score)
        self.mode = mode                # Variantes de GD: 'batch', 'sgd' o 'mbgd'
        self.batch_size = batch_size    # Tamaño de la muestra para Mini-Batch

        self._theta = np.array([])          # Coeficientes finales (beta)
        self._theta_scaled = np.array([])   # Coeficientes calculados sobre datos normalizados
        self._trained = False               # Estado del modelo

        self._mean = np.array([])           # Media para des-normalización
        self._std = np.array([])            # Desvío para des-normalización

        self.cost_history: List[float] = [] # Registro de la función de pérdida J(theta)

    def check_X_y(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Asegura la integridad de los datos de entrada y coincidencia de dimensiones."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X e y deben ser tensores de tipo np.ndarray")
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz de diseño (n_muestras, n_features)")
        if y.ndim != 1:
            raise ValueError("y debe ser un vector de etiquetas de 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistencia: X e y deben tener el mismo número de observaciones")
        return X, y

    def fit_standardize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforma las features para tener media 0 y varianza 1.
        Esencial para que el Descenso por Gradiente no oscile en features con distintas escalas.
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0  # Manejo de casos donde la varianza es nula para evitar NaN
        return (X - mean) / std, mean, std

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Agrega la columna de bias (unos) para permitir el cálculo del intercepto (theta_0)."""
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Calcula la Función de Costo J(theta) mediante la Suma de Cuadrados de los Residuos (RSS).
        Implementación del Error Cuadrático Medio (MSE) escalado.
        """
        n = X.shape[0]
        residuals = X @ theta - y  # Error de predicción: (y_hat - y)
        return (1.0 / (2 * n)) * (residuals @ residuals) # Producto punto para calcular suma de cuadrados

    def gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        learning_rate: float,
        iterations: int,
        tol: float,
        mode: str = "batch",
        batch_size: int = 32,
        random_state: int = 42
    ) -> Tuple[np.ndarray, int, List[float]]:
        """
        Optimización de pesos mediante el gradiente de la función de costo.
        Busca el mínimo global de la superficie de error J(theta).
        """
        n = X.shape[0]
        rng = np.random.default_rng(random_state) # Generador aleatorio para reproducibilidad
        cost_history: List[float] = []

        for i in range(iterations):
            old_theta = theta.copy() # Almacena estado previo para evaluar convergencia

            # --- Selección del método de actualización de pesos ---
            if mode == "batch":
                # Gradiente basado en el dataset completo (dirección óptima exacta)
                grad = (1/n) * (X.T @ (X @ theta - y))
            elif mode == "sgd":
                # Gradiente basado en una sola observación (estocástico, alta velocidad)
                idx = rng.integers(0, n)
                xi = X[idx]
                yi = y[idx]
                grad = xi * (xi @ theta - yi)
            elif mode == "mbgd":
                # Gradiente basado en un subconjunto (lote) de datos (eficiencia computacional)
                idx = rng.choice(n, size=min(batch_size, n), replace=False)
                Xb = X[idx]
                yb = y[idx]
                grad = (1/len(idx)) * (Xb.T @ (Xb @ theta - yb))
            else:
                raise ValueError("Modo de optimización no reconocido")

            # Actualización simultánea de los parámetros (regla de actualización de GD)
            theta = theta - learning_rate * grad
            cost_history.append(self.compute_cost(X, y, theta)) # Tracking de la pérdida

            # Criterio de parada: si la norma del cambio es menor a tol, el modelo convergió
            if np.linalg.norm(theta - old_theta) <= tol:
                return theta, i + 1, cost_history

        return theta, iterations, cost_history

    def _recover_original_theta(self) -> None:
        """
        Des-normalización de coeficientes: transforma theta de vuelta al espacio original 
        para que el usuario pueda usar datos crudos en predict().
        """
        if not self.normalize:
            self._theta = self._theta_scaled.copy()
            return

        theta0 = self._theta_scaled[0]
        rest = self._theta_scaled[1:]
        theta_rest = rest / self._std # Ajuste de pesos por el desvío estándar
        # Ajuste del intercepto compensando el desplazamiento de las medias
        theta0 = theta0 - np.sum((rest * self._mean) / self._std)
        self._theta = np.concatenate(([theta0], theta_rest))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo a los datos (Entrenamiento).
        Decide entre solución cerrada o aproximación iterativa.
        """
        X, y = self.check_X_y(X, y) # Validación de entrada
        if self.normalize:
            X, self._mean, self._std = self.fit_standardize(X) # Pre-procesamiento

        X = self._add_intercept(X) # Inclusión del bias

        if self.method == "analytic":
            # Solución por Mínimos Cuadrados Ordinarios (OLS) usando Pseudoinversa de Moore-Penrose
            self._theta_scaled = np.linalg.pinv(X) @ y
            self.cost_history = []
        elif self.method == "gd":
            # Optimización iterativa mediante Descenso por Gradiente
            theta0 = np.zeros(X.shape[1]) # Inicialización de pesos en cero
            self._theta_scaled, _, self.cost_history = self.gradient_descent(
                X, y, theta0, self.learning_rate, self.iterations, self.tol,
                mode=self.mode, batch_size=self.batch_size
            )
        else:
            raise ValueError("Método de entrenamiento inválido")

        self._recover_original_theta() # Regreso a escala original
        self._trained = True

    @property
    def theta(self) -> np.ndarray:
        """Retorna los parámetros aprendidos por el modelo (intercepción + coeficientes)."""
        if not self._trained:
            raise RuntimeError("Error: El modelo debe entrenarse con fit() antes de consultar theta")
        return self._theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Aplica la hipótesis aprendida H(X) = X @ theta para predecir nuevas instancias."""
        if not self._trained:
            raise RuntimeError("Error: El modelo debe entrenarse con fit() antes de realizar predicciones")
        X = self._add_intercept(X) # Agregado de bias a los datos de entrada
        return X @ self._theta # Producto escalar entre matriz de datos y vector de pesos