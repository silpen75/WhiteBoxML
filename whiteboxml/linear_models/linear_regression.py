# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import List, Tuple

class LinearRegression():
    """
    Modelo de regresión lineal con soporte para:
    - solución analítica
    - GD batch, SGD y mini-batch
    :author: Equipo Regresión Lineal
    :date: mayo 2026
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
        Inicializa el modelo.

        :param method: Método de entrenamiento ("analytic" o "gd").
        :param learning_rate: Tasa de aprendizaje (solo GD).
        :param iterations: Máximo número de iteraciones (solo GD).
        :param tol: Tolerancia de convergencia (solo GD).
        :param normalize: Si True, las variables de entrada se normalizan.
        :param mode: Modo de GD: "batch", "sgd" o "mbgd"
        :param batch_size: Tamaño del lote para el modo minibatch
        """

        self.method = method
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tol = tol
        self.normalize = normalize
        self.mode = mode
        self.batch_size = batch_size

        self._theta = np.array([])
        self._theta_scaled = np.array([])
        self._trained = False

        self._mean = np.array([])
        self._std = np.array([])

        self.cost_history: List[float] = []

    def check_X_y(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X e y deben ser np.ndarray")
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D")
        if y.ndim != 1:
            raise ValueError("y debe ser un vector 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de filas")
        return X, y

    def fit_standardize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std, mean, std

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        n = X.shape[0]
        residuals = X @ theta - y
        return (1.0 / (2 * n)) * (residuals @ residuals)

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
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        cost_history: List[float] = []

        for i in range(iterations):
            old_theta = theta.copy()

            if mode == "batch":
                grad = (1/n) * (X.T @ (X @ theta - y))
            elif mode == "sgd":
                idx = rng.integers(0, n)
                xi = X[idx]
                yi = y[idx]
                grad = xi * (xi @ theta - yi)
            elif mode == "mbgd":
                idx = rng.choice(n, size=min(batch_size, n), replace=False)
                Xb = X[idx]
                yb = y[idx]
                grad = (1/len(idx)) * (Xb.T @ (Xb @ theta - yb))
            else:
                raise ValueError("Modo inválido")

            theta = theta - learning_rate * grad
            cost_history.append(self.compute_cost(X, y, theta))

            if np.linalg.norm(theta - old_theta) <= tol:
                return theta, i + 1, cost_history

        return theta, iterations, cost_history

    def _recover_original_theta(self) -> None:
        if not self.normalize:
            self._theta = self._theta_scaled.copy()
            return

        theta0 = self._theta_scaled[0]
        rest = self._theta_scaled[1:]
        theta_rest = rest / self._std
        theta0 = theta0 - np.sum((rest * self._mean) / self._std)
        self._theta = np.concatenate(([theta0], theta_rest))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self.check_X_y(X, y)
        if self.normalize:
            X, self._mean, self._std = self.fit_standardize(X)

        X = self._add_intercept(X)

        if self.method == "analytic":
            self._theta_scaled = np.linalg.pinv(X) @ y
            self.cost_history = []
        elif self.method == "gd":
            theta0 = np.zeros(X.shape[1])
            self._theta_scaled, _, self.cost_history = self.gradient_descent(
                X, y,
                theta0,
                self.learning_rate,
                self.iterations,
                self.tol,
                mode=self.mode,
                batch_size=self.batch_size
            )
        else:
            raise ValueError("Método inválido")

        self._recover_original_theta()
        self._trained = True

    @property
    def theta(self) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Modelo no entrenado")
        return self._theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Modelo no entrenado")
        X = self._add_intercept(X)
        return X @ self._theta