class ModelHandler:
    def __init__(self,
                 seird_model,
                 date_of_first_infection,
                 dates,
                 Y_0):
        self.seird_model = seird_model
        self.fitting_params = self._get_fitting_params_from_model()
        self.dates = dates
        self.Y_0 = Y_0
        
    def get_fitting_params_from_model(self):
        """Sets how fitting parameters are related to the model attributes.

        Should be altered each time a new number of parameters is set, e.g. changing the number of groups or mixing environments. 
        """
        return [self.date_of_first_infection,
                self.seird_model.beta_vector[0],
                self.seird_model.beta_vector[1],
                self.seird_model.gamma,
                self.seird_model.delta[0],
                self.seird_model.theta_matrix[0,0],
                self.seird_model.theta_matrix[0,1]]
    
    def set_fitting_params_for_model(self, params):
        """Setting how the models parameters change.
        
        Should be altered each time a new number of parameters is changed, or the number of mixing environments, etc.
        """
        self.date_of_first_infection  = params[0]
        self.seird_model.beta_vector  = [params[1], params[2]]
        self.seird_model.gamma        = params[3]
        self.seird_model.delta        = params[4]*np.ones(self.seird_model.num_groups)
        self.seird_model.theta_matrix = fitting_deaths.build_theta_matrix(self.theta_matrix_building_method, params[5], params[6])

    def get_derivs_per_day(self):
        return self.seird_model.compute_derivs_per_day(self.dates,
                                                       self.date_of_first_infection,
                                                       self.Y_0)
    
    @staticmethod
    def build_theta_matrix(*args) -> npt.NDArray:
        """Builds the theta matrix from a series of arguments. 
    
        Args:
          If first argument is 'Default', the args should be
          the parameters that go into the theta matrix.
          Otherwise args[0], then it should be a function to call and pass the rest of the arguments to. Can be an externally defined function, held as self.theta_matrix_building_method. 
    
        Returns:
          theta_matrix (npt.NDArray): 
            The matrix $\theta^i_g$ that couples population groups $g$ to mixing environments $i$. 
            Should be shape $M\times N$, where $M$ is the number of mixing environments and $N$ the number of pop groups. 
        """
        theta_matrix = np.array([[   args[0],    args[1],    args[0]],
                                 [1.-args[0], 1.-args[1], 1.-args[0]]])
        return theta_matrix
