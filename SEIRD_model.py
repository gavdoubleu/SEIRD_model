#!/usr/bin/env python
"""SEIRD model class

Class containing methods specific to the multi-group SEIRD model. 

Attributes:
  beta_vector (npt.ArrayLike):
    List of the transmission coefficients for each mixing environment.
  gamma (npt.ArrayLike):
    Recovery rate. Scalar or length N array, where N is the number of population groups.
  delta (npt.ArrayLike):
    Death rate. Scalar or length N array, where N is the number of population groups.
  latent_period (npt.ArrayLike):
    Latency period between exposure and beginning infectiousness.
    Scalar or length N array.
  theta_matrix (npt.NDArray):
    The coupling coefficient between groups $g$ and mixing environments $m$.
    M x N matrix where M is the number of mixing environments, and N is the number of groups.
    All columns should sum to 1 as population groups must go somewhere.
  compartment_list (list[str], optional):
    list of names for each comparment. Used for plot labels. 
  group_names (list[str], optional):
    list of the names for each type of group. Used for plot labels. 

Methods:
  deriv_matrix:
    Finds the derivatives dY/dt for all compartments in the SEIRD model.
  compute_derivs_per_day:
    Compute the deriv_matrix dY/dt for a series of dates.
  compute_Y_t:
    Returns Y as a function of t for a series of dates.
  plot_Y_t:
    Plot the population of various compartments. 
"""

__author__      = "Gavin Woolman"
__copyright__   = "Copyright Sept 2025, Planet Earth"

import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SEIRD_model:
    def __init__(self,
                 beta_vector,
                 gamma,
                 delta,
                 latent_period,
                 theta_matrix,
                 compartment_list=['Susceptible',
                                   'Exposed',
                                   'Infectious',
                                   'Recovered',
                                   'Dead'],
                 group_names=['0-19','20-49','50+']):
        """Creates an instance of the SEIRD model class.

        Args:
          beta_vector (npt.ArrayLike):
            The transmission rate for each mixing environment. Length M where M is the number of mixing environments. 
          gamma (npt.ArrayLike):
            The recovery rate for each group. 
          delta (npt.ArrayLike): 
            The death rate for each group.
          latent_period (npt.ArrayLike):
            The mean length of time between exposure and becoming infectious for each group. 
          theta_matrix (np.ndarray):
            MxN array where M is the number of mixing environments, and N is the number of population groups. 
            Theta^i_j encodes the probability of each population type j going to mixing environment i. 
          compartment_list (List[str], optional):
            A list naming each compartment in order. 
        """
        self.beta_vector = beta_vector
        self.gamma = gamma
        self.delta = delta
        self.latent_period = latent_period
        self.theta_matrix = theta_matrix
        self.compartment_list = compartment_list
        self.group_names = group_names
        self.num_groups = self.theta_matrix.shape[1]
        self.num_compartments = len(self.compartment_list)
        self.num_mixing_envs = len(self.beta_vector)
        # Run some simple error checks
        self._simple_error_checks()

    def _simple_error_checks(self):
        """Performs some checks that the shape and content of various NDarrays are what they should be. 
        """
        if not self.theta_matrix.shape == (self.num_mixing_envs, self.num_groups):
            raise Exception("theta_matrix is not M by N, where M is the number of mixing environments = length of the beta_vector and N is the number of population groups: {} does not equal {}".format(self.theta_matrix.shape,(self.num_mixing_envs, self.num_groups)))
        if not np.array_equal(self.theta_matrix.sum(axis=0), np.ones(self.num_groups)):
            raise Exception("the theta_matrix isn't properly normalized (all columns should add to one)")

    # the SEIR model differential equations
    def deriv_matrix(self, t, Y: npt.NDArray) -> np.array:
        """
        Takes the parameters and state matrix $Y$ and computes the derivative dYdt for a given SEIRD model

        Args:
          t (npt.DTypeLike):
            Argument not used. Needed because solve_ivp passes the date on as the first argument explicitly. 
          Y (np.array): 
            The state matrix. This says the population in each compartment (S,I,R, and D), 
            for each type of population (e.g. old, young). Should be input as a flattened matrix.  
          self.beta_vector (np.array/list):
            The infection coefficient for each population's environment. Units of /day
            [beta_1, beta_2, ..., beta_M] where M is the number of types of mixing environment (e.g. party, non-party).
          self.gamma (npt.ArrayLike):
            the recovery rate in units of recoveries/day for each group.
            Scalar or arraylike of length N=number of groups. 
          self.delta (npt.ArrayLike):
            the death rate in units of deaths/day for each group.
            Scalar or arraylike of length N=number of groups. 
          self.theta_matrix (npt.NDArray):
            MxN matrix, where M is the number of mixing environments and N is the number of types of population
            The matrix encodes the portion of people of type i who visit environment j in the timestep. 
            can be thought of as a probability. Probability is independent of the compartment state (S,I,and R all visit),
            Dead folk don't visit. 
          self.latent_period (npt.ArrayLike):
            the average length of time in days between exposure and the beginning of infectiousness for each group. 
            Fixed at 1.4 days. 

        Returns:
          dYdt (np.array): 
            A flattened array of the same dimensions as Y, corresponding to dY/dt (the change in Y for 1 day). 
        """

        Y_matrix = Y.reshape(int(Y.size/self.num_compartments), self.num_compartments)

        S, E, I, R, D = Y_matrix.T

        B_env_matrix = np.matmul(self.theta_matrix, Y_matrix)

        S_env, E_env, I_env, R_env, D_env = B_env_matrix.T

        N_active_env = S_env + E_env + I_env + R_env  # just adds S, I, and R

        N_active_env[np.isclose(N_active_env,0)]=1
        # The vulnerable matrix ---- says which

        dYdt = np.zeros(Y_matrix.shape)

        new_infections_by_pop = np.zeros(len(S))

        #for i, s_i in enumerate(S):
        #    new_infections_by_pop[i] = np.matmul(s_i * theta_matrix.T[i] , beta_vector * I_env / N_active_env)
        new_infections_by_pop = np.matmul( np.multiply(S, self.theta_matrix).T , self.beta_vector * I_env / N_active_env)

        dYdt = np.array([-new_infections_by_pop,
                         new_infections_by_pop - E / self.latent_period,
                         E / self.latent_period - I*(self.gamma+self.delta), 
                         self.gamma*I,  
                         self.delta*I]).T
        
        return dYdt.flatten()


    def compute_Y_t(self,
                    dates: npt.ArrayLike,
                    Y_0: npt.NDArray) -> npt.NDArray:
        """Returns Y as a function of t for a series of dates.

        Args:
          dates (npt.ArrayLike):
            Array of dates on which to evaluate Y(t).
          Y_0 (npt.NDArray):
            Y at t=0.

        Returns:
          Y_t (npt.NDArray):
            Y as a function of t, where Y(t=0) = Y_0. Shape is (N , C, len(dates)).
        """
        # Now we use scipy to solve the system of differential equations
        solmat = solve_ivp(self.deriv_matrix, 
                       [min(dates), max(dates)], 
                       Y_0.flatten(), 
                       dense_output=True)
        
        Y_t = solmat.sol(dates).reshape(*Y_0.shape,len(dates))
        return Y_t
    
    
    def compute_derivs_per_day(self,
                               dates: npt.ArrayLike,
                               date_of_first_infection: float,
                               Y_0: npt.NDArray) -> npt.NDArray:
        """Compute the derivatives dY/dt for a series of dates.

        Args:
          dates (npt.ArrayLike):
            The days on which to evaluate the rate of change of the compartments. 
          date_of_first_infection (float): 
            The date on which infected individuals are first introduced. 
          Y_0 (np.ndarray):
            The distribution of population across all compartments
            on the first day of infection. 
          self.beta_vector (npt.ArrayLike):
            The transmission rate for each mixing environment.
            Length M where M is the number of mixing environments. 
          self.gamma (npt.ArrayLike):
            The recovery rate for each group. 
          self.delta (npt.ArrayLike): 
            The death rate for each group.
          self.latent_period (npt.ArrayLike):
            The mean length of time between exposure and becoming infectious
            for each group. 
          self.theta_matrix (np.ndarray):
            MxN array where M is the number of mixing environments,
            and N is the number of population groups. 
            Theta^i_j encodes the probability of each population type j
            going to mixing environment i. 

        Returns:
          derivs_for_all_days (np.ndarray):
            Derivatives for the compartments across a series of days. 
            An array with the same number of elements as the compartments multiplied 
            by the number of days. (Nx5xT) where N is the number of population groups, 
            5 is the number of compartments (S,E,I,R,& D), and T is the number of dates. 
        """

        if not Y_0.shape == (self.num_groups, self.num_compartments):
            raise Exception("Y_0 shape {} is not compatible with the number of groups {} and compartments {} in SEIRD model".format(Y_0.shape, self.num_groups, self.num_compartments))
        
        # shifts it so day zero is date_of_first_infection
        dates_after_start = dates[dates >= date_of_first_infection]-date_of_first_infection
        dates_before_start = dates[dates < date_of_first_infection]

        Y_tT = self.compute_Y_t(dates_after_start, Y_0).transpose((2,0,1))
        
        derivs_post_day_0 = np.zeros(Y_tT.shape)
        for i, y_day in enumerate(Y_tT):
            derivs_post_day_0[i] = self.deriv_matrix(dates_after_start, y_day).reshape(Y_0.shape)

        derivs_for_all_days = np.concatenate([np.zeros((len(dates_before_start),*Y_0.shape)), np.array(derivs_post_day_0)],
                                             axis=0).transpose((1,2,0))    

        return derivs_for_all_days # Return only the death data

    
    def plot_Y_t(self,
                 t,
                 Y_matrix_0,
                 groups_to_plot='all',
                 compartments_to_plot='all'):
        """Plot some compartments for a series of days $t$

        Args:
          t (npt.ArrayLike):
            The days over which to plot Y(t)
          Y_matrix_0 (npt.NDArray):
            The population distribution across groups and compartments on the first day of infection.
          groups_to_plot (List, optional):
            Indices of the groups to plot. Default is 'all', which plots all groups.
          compartments_to_plot (List, optional):
            Indices of the compartments to plot. Default is 'all', which plots them all.
            """
        if groups_to_plot == 'all':
            groups_to_plot = [i for i in range(self.num_groups)]
        if compartments_to_plot == 'all':
            compartments_to_plot = [i for i in range(self.num_compartments)]
        
        Y_t = self.compute_Y_t(t, Y_matrix_0)
        
        color_list = color=['b','olive','r','green','k']
        linestyle_list = ['solid', 'dashed', 'dotted', 'dashdot','(0,(1, 1))']
        
        plt.figure(figsize=(12,5))
        for group in groups_to_plot:
            for comp in compartments_to_plot:
                plt.plot(t,
                         Y_t[group, comp],
                         color=color_list[comp % Y_matrix_0.shape[1]],
                         linestyle=linestyle_list[group % Y_matrix_0.shape[0]],
                         label='{}, {}'.format(self.compartment_list[comp], self.group_names[group]),
                         alpha=.8)
        
        plt.title("SEIR Model Dynamics")
        plt.xlabel("Time (days)")
        plt.ylabel("Population in Compartment")
        plt.legend()
        plt.grid(lw=1,ls=":")
    
