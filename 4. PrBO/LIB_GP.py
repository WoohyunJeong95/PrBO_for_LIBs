import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score

class LIB_GP:
    def __init__(self,kernel_type, alpha = 1e-10, n_restarts = 100, random_state = 0, normalize_y = False):
        if kernel_type == 'RBF':
            self.kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        elif kernel_type == 'Matern32':
            self.kernel = 1 * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e3))
        elif kernel_type == 'Matern52':
            self.kernel = 1 * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-3, 1e3))

        self.n_restarts = n_restarts
        self.random_state = random_state
        self.normalize_y = normalize_y
        self.alpha = alpha
        self.model = GaussianProcessRegressor(kernel = self.kernel, alpha = self.alpha, n_restarts_optimizer=self.n_restarts, random_state = self.random_state, normalize_y = self.normalize_y)

    def GP_fit(self,X,y):
        self.model.fit(X,y)
        return self.model

    def GP_predict(self,model,X):
        mu, std = model.predict(X,return_std = True)
        return mu, std

    def GP_leave_one_out_cv(self,X_train,y_train,kernel_opt):
        R2score = np.array([])
        for j in range(len(X_train)):
            X_train_temp = np.delete(X_train, j, axis=0)
            y_train_temp = np.delete(y_train, j, axis=0)
            GP = GaussianProcessRegressor(kernel=kernel_opt, optimizer=None, random_state=0, normalize_y=self.normalize_y)
            GP.fit(X_train_temp, y_train_temp)
            mean_prediction, std_prediction = GP.predict(X_train, return_std=True)

            R2score = np.append(R2score, r2_score(y_train, mean_prediction))
        return np.mean(R2score)

    def GP_leave_one_out_cv_PrBO(self,X_train,y_train,kernel_opt,y_ECM,y):
        R2score = np.array([])
        for j in range(len(X_train)):
            X_train_temp = np.delete(X_train, j, axis=0)
            y_train_temp = np.delete(y_train, j, axis=0)
            GP = GaussianProcessRegressor(kernel=kernel_opt, optimizer=None, random_state=0, normalize_y=self.normalize_y)
            GP.fit(X_train_temp, y_train_temp)
            mean_prediction, std_prediction = GP.predict(X_train, return_std=True)

            R2score = np.append(R2score, r2_score(y, mean_prediction+y_ECM))
        return np.mean(R2score)