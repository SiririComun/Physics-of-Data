import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.svm import SVC
from scipy.stats import norm


class LearningTheoryEngine:
	"""Engine for synthetic binary system generation and analytical phase boundary."""

	def __init__(self, mu1=5.0, sigma1=2.5, mu2=7.5, sigma2=1.5):
		self.set_gaussian_parameters(mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)

	def set_gaussian_parameters(self, mu1, sigma1, mu2, sigma2):
		"""Defines parameters for two Gaussian states."""
		if sigma1 <= 0 or sigma2 <= 0:
			raise ValueError("Standard deviations must be positive.")

		self.mu1 = float(mu1)
		self.sigma1 = float(sigma1)
		self.mu2 = float(mu2)
		self.sigma2 = float(sigma2)
		return self.get_gaussian_parameters()

	def get_gaussian_parameters(self):
		return {
			"mu1": self.mu1,
			"sigma1": self.sigma1,
			"mu2": self.mu2,
			"sigma2": self.sigma2,
		}

	def _boundary_quadratic_coefficients(self):
		"""Returns coefficients (a, b, c) for the equality p1(x) = p2(x)."""
		inv_s1_sq = 1.0 / (self.sigma1 ** 2)
		inv_s2_sq = 1.0 / (self.sigma2 ** 2)

		a = inv_s1_sq - inv_s2_sq
		b = -2.0 * self.mu1 * inv_s1_sq + 2.0 * self.mu2 * inv_s2_sq
		c = (
			(self.mu1 ** 2) * inv_s1_sq
			- (self.mu2 ** 2) * inv_s2_sq
			+ 2.0 * np.log(self.sigma1 / self.sigma2)
		)
		return a, b, c

	def analytical_boundary(self):
		"""Calculates the analytical intersection point(s) x* where both likelihoods are equal."""
		a, b, c = self._boundary_quadratic_coefficients()

		if np.isclose(a, 0.0):
			if np.isclose(b, 0.0):
				raise ValueError("Degenerate system: unable to determine a unique boundary.")
			roots = np.array([-c / b], dtype=float)
		else:
			discriminant = b ** 2 - 4.0 * a * c
			if discriminant < 0:
				raise ValueError("No real analytical boundary found for the current parameters.")
			sqrt_discriminant = np.sqrt(discriminant)
			roots = np.array(
				[(-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)],
				dtype=float,
			)

		low_mu = min(self.mu1, self.mu2)
		high_mu = max(self.mu1, self.mu2)
		between_mask = (roots >= low_mu) & (roots <= high_mu)

		if between_mask.any():
			selected_root = float(np.sort(roots[between_mask])[0])
		else:
			center = 0.5 * (self.mu1 + self.mu2)
			selected_root = float(roots[np.argmin(np.abs(roots - center))])

		return {
			"boundary": selected_root,
			"roots": [float(root) for root in np.sort(roots)],
			"coefficients": {"a": float(a), "b": float(b), "c": float(c)},
		}

	def generate_training_samples(self, n_samples=100, random_state=42):
		"""Generates random variates for both Gaussian states."""
		if n_samples <= 0:
			raise ValueError("n_samples must be a positive integer.")

		rng = np.random.default_rng(random_state)
		samples_state_0 = norm.rvs(
			loc=self.mu1,
			scale=self.sigma1,
			size=int(n_samples),
			random_state=rng,
		)
		samples_state_1 = norm.rvs(
			loc=self.mu2,
			scale=self.sigma2,
			size=int(n_samples),
			random_state=rng,
		)
		return samples_state_0, samples_state_1

	def generate_test_set(self, n_samples=50, random_state=2027):
		"""Generates a labeled test DataFrame with keys Y and X_1."""
		samples_state_0, samples_state_1 = self.generate_training_samples(
			n_samples=n_samples,
			random_state=random_state,
		)
		return self.build_labeled_dataframe(samples_state_0, samples_state_1)

	@staticmethod
	def build_labeled_dataframe(samples_state_0, samples_state_1):
		"""Builds the training DataFrame with columns Y and X_1."""
		feature_values = np.concatenate([samples_state_0, samples_state_1])
		label_values = np.concatenate(
			[
				np.zeros(len(samples_state_0), dtype=int),
				np.ones(len(samples_state_1), dtype=int),
			]
		)

		return pd.DataFrame({"Y": label_values, "X_1": feature_values})

	def save_synthetic_system(
		self,
		boundary_report,
		filename="03_synthetic_system.json",
		model_scores=None,
		test_configuration=None,
	):
		"""Persists Gaussian parameters and boundary report as a JSON artifact."""
		path = os.path.join("..", "artifacts", filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)

		payload = {
			"gaussian_parameters": self.get_gaussian_parameters(),
			"analytical_boundary": boundary_report,
		}
		if test_configuration is not None:
			payload["test_configuration"] = test_configuration
		if model_scores is not None:
			payload["model_scores"] = model_scores

		with open(path, "w", encoding="utf-8") as file:
			json.dump(payload, file, indent=4, ensure_ascii=False)

		return path

	def plot_learning_curves(
		self,
		estimator,
		features,
		labels,
		filename,
		cv=5,
		train_size_grid=None,
		scoring="accuracy",
		random_state=42,
	):
		"""Computes and plots learning curves, then saves the figure into artifacts/."""
		if train_size_grid is None:
			train_size_grid = np.linspace(0.2, 1.0, 6)

		cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
		train_sizes, train_scores, valid_scores = learning_curve(
			estimator,
			features,
			labels,
			train_sizes=train_size_grid,
			cv=cv_splitter,
			scoring=scoring,
			n_jobs=None,
		)

		train_mean = train_scores.mean(axis=1)
		train_std = train_scores.std(axis=1)
		valid_mean = valid_scores.mean(axis=1)
		valid_std = valid_scores.std(axis=1)

		plt.figure(figsize=(9, 5))
		plt.plot(train_sizes, train_mean, marker="o", color="#2a9d8f", label="Training score")
		plt.plot(train_sizes, valid_mean, marker="s", color="#e76f51", label="Validation score")
		plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="#2a9d8f")
		plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2, color="#e76f51")
		plt.title("Learning Curve")
		plt.xlabel("Training Samples")
		plt.ylabel(scoring.capitalize())
		plt.legend()
		plt.grid(alpha=0.25)

		path = os.path.join("..", "artifacts", filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		plt.tight_layout()
		plt.savefig(path, bbox_inches="tight")
		plt.close()

		return {
			"path": path,
			"train_sizes": [int(size) for size in train_sizes],
			"train_mean": [float(value) for value in train_mean],
			"validation_mean": [float(value) for value in valid_mean],
		}

	@staticmethod
	def save_model_optimization(payload, filename="03_model_optimization.json"):
		"""Persists hyperparameter optimization results as a JSON artifact."""
		path = os.path.join("..", "artifacts", filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)

		with open(path, "w", encoding="utf-8") as file:
			json.dump(payload, file, indent=4, ensure_ascii=False)

		return path

	def gamma_sweep(
		self,
		features,
		labels,
		gamma_values=None,
		test_size=0.2,
		validation_fraction=0.125,
		random_state=42,
	):
		"""Sweeps gamma for an RBF SVC and returns train/validation/test score traces."""
		if gamma_values is None:
			gamma_values = np.logspace(-3, 3, 25)

		x_train, x_test, y_train, y_test = train_test_split(
			features,
			labels,
			test_size=test_size,
			stratify=labels,
			random_state=random_state,
		)

		train_scores = []
		validation_scores = []
		test_scores = []

		n_splits = max(2, int(round(1.0 / validation_fraction)))
		cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

		for gamma in gamma_values:
			model = SVC(kernel="rbf", gamma=float(gamma), C=1.0)
			model.fit(x_train, y_train)

			train_score = model.score(x_train, y_train)
			validation_score = np.mean(cross_val_score(model, x_train, y_train, cv=cv_splitter, scoring="accuracy"))
			test_score = model.score(x_test, y_test)

			train_scores.append(float(train_score))
			validation_scores.append(float(validation_score))
			test_scores.append(float(test_score))

		best_index = int(np.argmax(validation_scores))
		best_gamma = float(gamma_values[best_index])

		return {
			"gamma_values": [float(gamma) for gamma in gamma_values],
			"train_scores": train_scores,
			"validation_scores": validation_scores,
			"test_scores": test_scores,
			"best_gamma": best_gamma,
			"best_validation_score": float(validation_scores[best_index]),
			"best_test_score": float(test_scores[best_index]),
			"split": {
				"test_size": float(test_size),
				"train_size": float(1.0 - test_size),
				"validation_fraction_within_train": float(validation_fraction),
			},
		}

	@staticmethod
	def save_model_selection(payload, filename="03_model_selection.json"):
		"""Persists model-selection and phase-transition diagnostics as JSON artifact."""
		path = os.path.join("..", "artifacts", filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)

		with open(path, "w", encoding="utf-8") as file:
			json.dump(payload, file, indent=4, ensure_ascii=False)

		return path
