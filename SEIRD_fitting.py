#!/usr/bin/env python
"""SEIRD fitting class

Class designed to fit an already created SEIRD model to some data. 
"""

__author__      = "Gavin Woolman"
__copyright__   = "Copyright Sept 2025, Planet Earth"

import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class fitting_deaths:
    def __init__(self,
                 dates: npt.ArrayLike,
                 y_data: npt.ArrayLike,
                 sigma_y: npt.ArrayLike,
                 seird_model: "SEIRD_model",
                 bounds: list[tuple],
                 Y_0: npt.NDArray,
                day_zero: float = None):
        self.dates            = dates
        self.y_data           = y_data
        self.sigma_y          = sigma_y
        self.seird_model      = seird_model
        self.initial_model    = seird_model
        self.Y_0              = Y_0
        self.Yindex_of_y_data = 4
        self.bounds           = bounds
        if day_zero == None:
            self.date_of_first_infection = min(self.dates)
        else:
            self.date_of_first_infection = day_zero
        self._get_fitting_params_from_model()

    def _get_fitting_params_from_model(self):
        self.fitting_params = [self.date_of_first_infection,
                               self.seird_model.beta_vector[0], 
                               self.seird_model.beta_vector[1], 
                               self.seird_model.gamma, 
                               self.seird_model.theta_matrix[0,0],
                              self.seird_model.theta_matrix[0,1]]
    
    def _set_fitting_params_for_model(self, params):
        self.date_of_first_infection  = params[0]
        self.seird_model.beta_vector  = [params[1], params[2]]
        self.seird_model.gamma        = params[3]
        self.seird_model.delta        = params[3] * 0.1 * np.array([1.,1.,2.])
        self.seird_model.theta_matrix = fitting_deaths.build_theta_matrix(params[4], params[5])


    def _error_function(self) -> float:
        """Computes the xi^2 error for the death-data fit. 
        """
        derivs_for_all_days = self.seird_model.compute_derivs_per_day(self.dates,
                                                                      self.date_of_first_infection,
                                                                      self.Y_0)
        self.y_fit = derivs_for_all_days[:,self.Yindex_of_y_data,:]
        self.residuals_squared = (self.y_fit - self.y_data)*(self.y_fit - self.y_data) / (self.sigma_y*self.sigma_y) 
        self.xi = np.sqrt(np.sum(self.residuals_squared) / len(self.residuals_squared))
        return self.xi
    
    
    def _fun_to_minimize(self, new_params: npt.ArrayLike) -> float:
        self.fitting_params = new_params
        self._set_fitting_params_for_model(new_params)
        return self._error_function()


    def do_minimize(self, method='Nelder-Mead', options={'maxiter':3000}, tol=1.0e-12):
        self.opt = minimize(self._fun_to_minimize, 
                            self.fitting_params,
                            bounds=self.bounds,
                            tol=tol,
                            method=method,
                            options=options)



    @staticmethod
    def build_theta_matrix(*args) -> npt.NDArray:
        """Builds the theta matrix from a series of arguments. 
    
        Args:
          The parameters that go into the theta matrix. 
    
        Returns:
          theta_matrix (npt.NDArray): 
            The matrix $\theta^i_g$ that couples population groups $g$ to mixing environments $i$. 
            Should be shape $M\times N$, where $M$ is the number of mixing environments and $N$ the number of pop groups. 
        """
        theta_matrix = np.array([[   args[0],    args[1],    args[0]],
                                 [1.-args[0], 1.-args[1], 1.-args[0]]])
        return theta_matrix

    def plot_fit(self):
        """Plot the fit and residuals
        """
        self._error_function()
        print("Xi value = {}".format(self.xi))
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))

        ax1.scatter(self.dates, self.y_data[0], color='green',label='Sparse Young Deaths per Day')
        ax1.scatter(self.dates, self.y_data[1], color='blue',label='Sparse Old Deaths per Day')
        derivs = self.seird_model.compute_derivs_per_day(self.dates,
                                                         self.date_of_first_infection,
                                                         self.Y_0)
        ax1.plot(self.dates, derivs[0,self.Yindex_of_y_data,:], color='green', label='Young initial guess')
        ax1.plot(self.dates, derivs[1,self.Yindex_of_y_data,:], color='blue', label='Old initial guess')
        ax1.set_title('Fit'), ax1.set_xlabel('Day'), ax1.set_ylabel('Deaths per day')
        ax1.legend()
        residuals_0 = self.y_data[0] - derivs[0,self.Yindex_of_y_data,:]
        residuals_1 = self.y_data[1] - derivs[1,self.Yindex_of_y_data,:]
        ax2.scatter(self.dates, residuals_0, color='green', label='Young residuals')
        ax2.scatter(self.dates, residuals_1, color='blue', label='Old residuals')
        ax2.set_title('Residuals'), ax2.set_xlabel('Day'), ax2.set_ylabel('Residuals (deaths/day)')
        ax2.legend()
