import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


class LinearRegressionModel:

	def __init__(self, max_iterations, learning_rate, error_margin):
		self.max_iterations = max_iterations
		self.learning_rate = learning_rate
		self.error_margin = error_margin

	def fit(self, X, y):
		self.X = X
		self.y = y
		self.m = self.X.shape[0]

		self.X = np.array(self.X).reshape(-1, 1) 
		self.y = np.array(self.y).reshape(-1, 1)
		self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))

		self.betas = np.zeros((2,1))
		self.betas = np.array(self.betas).reshape(-1, 1)

		self.betas = self.gradient_steps()

		y_hat = self.betas[1]*self.X
		y_hat = y_hat + self.betas[0]

		return y_hat

	def loss(self, coeffs):
	    #y_hat = self.betas[1]*self.X
	    #y_hat = y_hat + self.betas[0]
	    #m = len(self.y) 

	    y_hat = coeffs[0] + coeffs[1] * self.X[:, 1]
	    #print("LOSS" + str(self.m))
	    E = (1/self.m) * np.sum(np.square(self.y - y_hat))
	    print(E)
	    return E


	def get_gradient(self, curr_step):

	    #self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
		curr_step = np.array(curr_step).reshape(-1, 1)

		m = self.X.shape[0]
	     
		gradient = (1/m) * (self.X.T @ (self.X @ curr_step - self.y))

		return gradient


	def gradient_steps(self):
	    iters = 0
	    grad_step = self.betas
	    grad_step = np.zeros((2, 1))
	    grad_step = grad_step - self.learning_rate*self.get_gradient(grad_step)
	    
	    while self.loss(grad_step) > self.error_margin and iters < self.max_iterations:
	        grad_step = grad_step - self.learning_rate*(self.get_gradient(grad_step))
	        #print(grad_step)
	        iters += 1

	    self.betas = grad_step
	    print("steps taken: " + str(iters))
	    print("loss: " + str(self.loss(grad_step)))

	    return self.betas



def main():

	np.random.seed(42)  # For reproducibility
	n_samples = 100

	# Generate X values (independent variable)
	X = np.linspace(0, 10, n_samples).reshape(-1, 1)

	# Define true relationship (y = 3X + 5 + noise)
	noise = np.random.normal(scale=2.0, size=(n_samples, 1))  # Add some noise
	y = 3 * X + 5 + noise  # Linear relationship with noise
	#betas = np.zeros((2,1))
	model = LinearRegressionModel(100000, 0.01, 3.5)

	# scaler = StandardScaler()
	# X_scaled = scaler.fit_transform(X)

	y_hat = model.fit(X, y)
	X_transformed = np.hstack((np.ones((X.shape[0], 1)), X))  # Ensure bias column
	y_hat = X_transformed @ model.betas 

	plt.figure(figsize=(8, 5))
	plt.scatter(X, y, label="Noisy Data", alpha=0.6, color="blue", edgecolor="black")
	plt.plot(X, y_hat, color="red", linewidth=2, label="Linear Regression Fit")

	# Labels and title
	plt.xlabel("X (Feature)")
	plt.ylabel("y (Target)")
	plt.title("Simulated Linear Data with Noise and Regression Fit")
	plt.legend()
	plt.grid(True)
	plt.show()


main()