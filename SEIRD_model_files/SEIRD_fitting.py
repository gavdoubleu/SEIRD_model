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

class DeathFitter:
    def __init__(self,
                 dates: npt.ArrayLike,
                 y_data: npt.ArrayLike,
                 sigma_y: npt.ArrayLike,
                 seird_model: "SEIRD_model",
                 bounds: list[tuple] = None,
                 model_handler = None):
        self.dates            = dates
        self.y_data           = y_data
        self.sigma_y          = sigma_y
        self.seird_model      = seird_model
        self.initial_model    = seird_model
        self.Yindex_of_y_data = 4
        self.bounds           = bounds
        self.model_handler = model_handler
        if model_handler is None:
            raise Exception("Need to set a model handler in order to do a fit.")
        else:
            self.fitting_params = self.model_handler.get_fitting_params_from_model()
        self.use_linear_constraint = False
        

    def _error_function(self) -> float:
        """Computes the xi^2 error for the death-data fit. 
        """
        derivs_for_all_days = self.model_handler.get_derivs_per_day()
        
        self.y_fit = derivs_for_all_days[:,self.Yindex_of_y_data,:]
        self.residuals_squared = (self.y_fit - self.y_data)*(self.y_fit - self.y_data) / (self.sigma_y*self.sigma_y) 
        self.xi = np.sqrt(np.sum(self.residuals_squared) / len(self.residuals_squared))
        return self.xi
    
    
    def _fun_to_minimize(self, new_params: npt.ArrayLike) -> float:
        # Record the new parameters
        self.fitting_params = new_params
        # Update the model
        self.model_handler.set_fitting_params_for_model(new_params)
        # Return the error metric
        return self._error_function()

    def do_minimize(self,
                    method='Nelder-Mead',
                    tol=1.0e-12,
                    **kwargs):
        """Minimizes the xi^2 to obtain the model that best fits the data."""
        if self.model_handler is None:
            raise Exception("Need to set self.model_handler object in order to fit")
        if self.use_constraints:
            self.opt = minimize(self._fun_to_minimize,
                                self.fitting_params,
                                constraints=self.constraints,
                                tol=tol,
                                method=method,
                                **kwargs)
        elif not (self.bounds is None):
            self.opt = minimize(self._fun_to_minimize,
                                self.fitting_params,
                                bounds=self.bounds,
                                tol=tol,
                                method=method,
                                **kwargs)
        else:
            self.opt = minimize(self._fun_to_minimize,
                                self.fitting_params,
                                tol=tol,
                                method=method,
                                **kwargs)

    def plot_fit(self,
                 labellist=None,
                 xplot=None,
                 plot_rate=False,
                 linewidth=3.,
                 alpha=.5,
                 color_list=None,
                 linestyle_list=None,
                 marker_list=None,
                 s=10,
                 **kwargs):
        """Plot the fit and residuals"""
        self._error_function()
        print("Xi value = {}".format(self.xi))
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))

        if color_list is None:
            color_list = color=['b','olive','r','green','k']
        if linestyle_list is None:
            linestyle_list = ['solid', 'dashed', 'dotted', 'dashdot','(0,(1, 1))']
        if marker_list is None:
            marker_list = ['o','s','^','*','P']
        
        if labellist is None:
            labellist = ['']
        if xplot is None:
            xplot = self.dates

        if plot_rate:

            rescale_y = 10000
            N_per_group = self.model_handler.Y_0.sum(axis=1)
            
            for i,y in enumerate(self.y_data):
                ax1.scatter(xplot,
                            y/N_per_group[i]*rescale_y,
                            color=color_list[i % len(color_list)],
                            marker=marker_list[i % len(marker_list)],
                            s=s,
                            label='Data {}'.format(labellist[i % len(labellist)]),
                            alpha=alpha,
                            **kwargs)

            derivs = self.model_handler.get_derivs_per_day()

            for i,y in enumerate(derivs):
                ax1.plot(xplot,
                         y[self.Yindex_of_y_data]/N_per_group[i]*rescale_y,
                         color=color_list[i % len(color_list)],
                         linestyle=linestyle_list[i % len(linestyle_list)],
                         linewidth=linewidth,
                         label='Fit {}'.format(labellist[i % len(labellist)]),
                         zorder=10,
                         **kwargs)

            ax1.set_title('Fit'), ax1.set_xlabel('Day'), ax1.set_ylabel('Deaths per day per 10,000 people')
            ax1.legend()

            for i,y in enumerate(self.y_data):
                residuals = y - derivs[i,self.Yindex_of_y_data]
                ax2.scatter(xplot,
                            residuals/N_per_group[i]*rescale_y,
                            color=color_list[i % len(color_list)],
                            marker=marker_list[i % len(marker_list)],
                            s=s,
                            label='{}'.format(labellist[i % len(labellist)]),
                            alpha=alpha,
                            **kwargs)

            ax2.set_title('Residuals (deaths / day / 10,000 people)'), ax2.set_xlabel('Day'), ax2.set_ylabel('Residuals (deaths/day)')
            ax2.legend()
            
        else:
            for i,y in enumerate(self.y_data):
                ax1.scatter(xplot,
                            y,
                            color=color_list[i % len(color_list)],
                            marker=marker_list[i % len(marker_list)],
                            s=s,
                            label='Data {}'.format(labellist[i % len(labellist)]),
                            alpha=alpha,
                            **kwargs)

            derivs = self.model_handler.get_derivs_per_day()

            for i,y in enumerate(derivs):
                ax1.plot(xplot,
                         y[self.Yindex_of_y_data],
                         color=color_list[i % len(color_list)],
                         linestyle=linestyle_list[i % len(linestyle_list)],
                         label='Fit {}'.format(labellist[i % len(labellist)]),
                         zorder=10,
                         **kwargs)

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
                            alpha=alpha,
                            **kwargs)

            ax2.set_title('Residuals'), ax2.set_xlabel('Day'), ax2.set_ylabel('Residuals (deaths/day)')
            ax2.legend()
