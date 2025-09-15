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
        self.seird_model.delta        = params[3] * 0.02 * np.array([1., 1., 2.])
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
        # Record the new parameters
        self.fitting_params = new_params
        # Update the model
        self._set_fitting_params_for_model(new_params)
        # Return the error metric
        return self._error_function()


    def do_minimize(self,
                    method='Nelder-Mead',
                    options={'maxiter':3000},
                    tol=1.0e-12):
        """Minimizes the xi^2 to obtain the model that best fits the data."""
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

    def plot_fit(self, labellist=None, xplot=None):
        """Plot the fit and residuals"""
        self._error_function()
        print("Xi value = {}".format(self.xi))
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))

        color_list = color=['b','olive','r','green','k']
        linestyle_list = ['solid', 'dashed', 'dotted', 'dashdot','(0,(1, 1))']
        marker_list = ['.','s','^','*','P']

        if labellist is None:
            labellist = ['']
        if xplot is None:
            xplot = self.dates

        for i,y in enumerate(self.y_data):
            ax1.scatter(xplot,
                        y,
                        color=color_list[i % len(color_list)],
                        marker=marker_list[i % len(marker_list)],
                        s=10,
                        label='Data {}'.format(labellist[i % len(labellist)]),
                        alpha=.7)
        
        derivs = self.seird_model.compute_derivs_per_day(self.dates,
                                                         self.date_of_first_infection,
                                                         self.Y_0)
        for i,y in enumerate(derivs):
            ax1.plot(xplot,
                     y[self.Yindex_of_y_data],
                     color=color_list[i % len(color_list)],
                     linestyle=linestyle_list[i % len(linestyle_list)],
                     label='Fit {}'.format(labellist[i % len(labellist)]),
                     zorder=10)
        
        ax1.set_title('Fit'), ax1.set_xlabel('Day'), ax1.set_ylabel('Deaths per day')
        ax1.legend()

        for i,y in enumerate(self.y_data):
            residuals = y - derivs[i,self.Yindex_of_y_data]
            ax2.scatter(xplot,
                        residuals,
                        color=color_list[i % len(color_list)],
                        marker=marker_list[i % len(marker_list)],
                        s=10,
                        label='{}'.format(labellist[i % len(labellist)]),
                        alpha=.8)
        
        ax2.set_title('Residuals'), ax2.set_xlabel('Day'), ax2.set_ylabel('Residuals (deaths/day)')
        ax2.legend()
