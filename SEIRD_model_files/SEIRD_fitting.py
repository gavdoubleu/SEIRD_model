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
from scipy.optimize import basinhopping

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
        self.use_constraints = False
        

    def error_function(self) -> float:
        """Computes the Chi2 error for the death-data fit.

        Minimizing Chi2 should be equivalent to maximising the likelihood of seeing the data given a specific model. 
        
        """
        derivs_for_all_days = self.model_handler.get_derivs_per_day()
        
        self.y_fit = derivs_for_all_days[:,self.Yindex_of_y_data,:]
        self.residuals_squared = (self.y_fit - self.y_data)*(self.y_fit - self.y_data) 
        self.chi2 = np.sum(self.residuals_squared / (self.sigma_y*self.sigma_y))
        return self.chi2
    
    
    def __fun_to_minimize(self, new_params: npt.ArrayLike) -> float:
        # Record the new parameters
        self.fitting_params = new_params
        # Update the model
        self.model_handler.set_fitting_params_for_model(new_params)
        # Return the error metric
        return self.error_function()

    def do_minimize(self,
                    method='Nelder-Mead',
                    tol=1.0e-12,
                    **kwargs):
        """Minimizes chi to obtain the model that best fits the data."""
        if self.model_handler is None:
            raise Exception("Need to set self.model_handler object in order to fit")
        if self.use_constraints:
            self.opt = minimize(self.__fun_to_minimize,
                                self.fitting_params,
                                constraints=self.constraints,
                                tol=tol,
                                method=method,
                                **kwargs)
        elif not (self.bounds is None):
            self.opt = minimize(self.__fun_to_minimize,
                                self.fitting_params,
                                bounds=self.bounds,
                                tol=tol,
                                method=method,
                                **kwargs)
        else:
            self.opt = minimize(self.__fun_to_minimize,
                                self.fitting_params,
                                tol=tol,
                                method=method,
                                **kwargs)


    def __save_basin_hop_iteration(self, x, f, accept):
        self.basin_local_minima.sort(key = lambda minima: minima[1])
        if len(self.basin_local_minima) < self.__num_minima_to_save:
            if accept:
                self.basin_local_minima += [(x, f, accept)]
        else:
            if accept:
                if f < self.basin_local_minima[-1][1]:
                    self.basin_local_minima[-1] = (x, f, accept)

    def __write_basin_to_file(self,filename,x,f,accept):
        with open(filename, 'a') as phile:
            phile.write("{}, {}, {}\n".format(x, f, accept))

        
    def __save_basin_hop_iteration_to_file(self, x, f, accept):
        self.basin_local_minima.sort(key = lambda minima: minima[1])
        if len(self.basin_local_minima) < self.__num_minima_to_save:
            if accept:
                self.basin_local_minima += [(x, f, accept)]
                self.__write_basin_to_file(self.save_filename, x, f, accept)
        else:
            if accept:
                if f < self.basin_local_minima[-1][1]:
                    self.basin_local_minima[-1] = (x, f, accept)
                    self.__write_basin_to_file(self.save_filename, x, f, accept)

    def __accept_test(self, f_new, x_new, f_old, x_old):
        if (f_new-f_old < 0) | (np.std(np.array(x_new)-np.array(x_old)) > self.__accept_param_difference_cutoff):
            return True
        else:
            return False

    @classmethod
    def write_basins_to_file(self, filename, basins):
        with open(filename, 'a') as phile:
            for basin in basins:
                phile.write("{}, {}, {}\n".format(*basin))

    def basin_hopping(self,
                      num_minima_to_save: int = 1,                      
                      niter: int=10,
                      param_difference_cutoff: float = 0.001,
                      minimizer_kwargs: any = None,
                      save_direct_to_file: bool = False,
                      save_filename: str=None,
                      **kwargs):
        """Do the basin_hopping algorithm to explore the different possible fits.

        Able to save accepted local minima to a file if desired.

        Args:
          num_minima_to_save (int, optional):
            The number of basin_hopping local minima to save in the list of my_fit.basin_local_minima. Default = 1. 
          niter (int, optional):
            Number of iterations to do the basin_hopping algorithm over. Default = 10.
          param_difference_cutoff (float, optional):
            minimum standard deviation between a new basin local minima and the old one in order to be accepted (stops finding lots of local minima close by to each other). Default = 0.001.
          minimizer_kwargs (any, optional):
            kwargs for the minimizer algorithm. Default = None.
          save_direct_to_file (bool):
            If True, writes basin local minima directly to a file as soon as they are accepted by __accept_test.
            This is so that, if the code breaks halfway through, local minima are still stored.
          save_filename (str, optional):
            Required only if save_direct_to_file is True. Path to the save file.
          **kwargs (any):
            kwargs for the scipy.optimize.basin_hopping method. 
          
        
        """
        
        if save_direct_to_file:
            if save_filename is None:
                raise ValueError("if save_direct_to_file = True, then the filename to save to (save_filename: str) must not be None")
            else:
                self.save_filename = save_filename
        
        # # Changing default kwargs for the minimizer
        if minimizer_kwargs is None:
            minimizer_kwargs = {'method'  : 'cobyqa',
                                'options' : {'maxiter':5000},
                                'tol'     : 1.0e-8 }
        if self.use_constraints:
            if not 'constraints' in minimizer_kwargs.keys():
                minimizer_kwargs['constraints'] = self.constraints
        elif not (self.bounds is None):
            if not 'bounds' in minimizer_kwargs.keys():
                minimizer_kwargs['bounds'] = self.bounds

        if not hasattr(self, 'basin_local_minima'):
            self.basin_local_minima = []
        self.__num_minima_to_save = num_minima_to_save
        self.__accept_param_difference_cutoff = param_difference_cutoff
        if save_direct_to_file:
            self.basin_opt = basinhopping(self.__fun_to_minimize,
                                      self.fitting_params,
                                      niter=niter,
                                      callback=self.__save_basin_hop_iteration_to_file,
                                      minimizer_kwargs=minimizer_kwargs,
                                      accept_test=self.__accept_test,
                                      **kwargs)
        else:
            self.basin_opt = basinhopping(self.__fun_to_minimize,
                                      self.fitting_params,
                                      niter=niter,
                                      callback=self.__save_basin_hop_iteration,
                                      minimizer_kwargs=minimizer_kwargs,
                                      accept_test=self.__accept_test,
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
        """Plot the fit and residuals.

        Able to plot the fit or the 'rates', where the deaths have been divided by the number of people in the category.

        """
        Chi_value = self.error_function()
        print("Chi^2 value = {}".format(Chi_value))
        
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
            ax1.tick_params(labelrotation=90, axis='x')
            ax1.tick_params(direction='in', axis='both', right=True, top=True)
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
            ax2.tick_params(labelrotation=90, axis='x')
            ax2.tick_params(direction='in', axis='both', right=True, top=True)
            ax2.set_title(r'Residuals, $\sigma$ = {:.2f}'.format(np.std(residuals/N_per_group[i]*rescale_y))), ax2.set_xlabel('Day'), ax2.set_ylabel('Residuals (deaths/day/10,000 people)')
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

            ax1.tick_params(labelrotation=90, axis='x')
            ax1.tick_params(direction='in', axis='both', right=True, top=True)            
            ax1.set_title(r'Fit. $\chi^2 =$ {:.2f}'.format(Chi_value)), ax1.set_xlabel('Day'), ax1.set_ylabel('Deaths per day')
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

            ax2.tick_params(direction='in', axis='both', right=True, top=True)            
            ax2.tick_params(labelrotation=90, axis='x')
            ax2.set_title(r'Residuals, $\sigma = {:.2f}$ deaths/day'.format(np.std(residuals))), ax2.set_xlabel('Day'), ax2.set_ylabel('Residuals (deaths/day)')
            ax2.legend()

    
    def params_to_model(self, params):
        """Updates self.model to have the parameters given by fitting_params.

        """
        self.model_handler.set_fitting_params_for_model(params)
