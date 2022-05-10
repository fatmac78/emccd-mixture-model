import math
from copy import deepcopy
import numpy as np
from scipy import stats

class SerialCicDistribution:

    """A distribution modelling the Serial CIC in the EMCCD

    The distibution consists of a sum of exponential PDFs, each scaled by a regisiter,
    modelling probabaility a single sCIC event could happen in any of the registers

    """
    def __init__(self, **kwargs):
            
        self.beta: float = kwargs.get('register_gain')
        self.register_count: float = kwargs.get('register_count')
        self.exp_loc: float = kwargs.get('exp_loc')

              
        self.serial_cic_distributions = []

        for register in range(0, self.register_count):
            scale=(self.beta*register)/self.register_count
            self.serial_cic_distributions.append(stats.expon(loc=self.exp_loc, 
                                            scale=scale)                                            )
            
    def compute_prob(self, x)-> float:       
        
        prob = 0                                            
        
        for register in range(1, self.register_count):
            dist_prob = self.serial_cic_distributions[register].pdf(x)
            prob = prob + dist_prob
        
        prob = prob/self.register_count
        
        return prob
        
    def pdf(self, x)-> float:
                  
        prob = self.compute_prob(x)    
        
        return prob
    
    def logpdf(self, x)-> float:
    
        prob = self.compute_prob(x)
        
        return np.log(prob)

    def sample(self, size=1):
        """Sample the sCIC distribution. Each value of size will produce\
           a sample count matching number of registers

        Returns:
            Samples from sCIC distribution
        """
        samples = []

        for multiple in range(0, size):
            for register in range(0, self.register_count):
                scale=(self.beta*register)/self.register_count
                data = np.random.exponential(scale = scale, size=1) + self.exp_loc
                samples.append(data[0])

        return samples

class EmccdModelParameters:
    """Holds the parameters used in GaussianExponentialMixture.

    This class allows for access to parameters by name, pretty-printing,
    and comparison to other parameters to check for convergence.

    Args:
        beta (float): the scale parameter and mean for the exponential
            distribution this also corresponds to the mean, or the
            inverse of the rate of the exponential distribution.
        mu (float): the location parameter and mean for the gaussian
            distribution.
        sigma (float): the scale parameter and the standard deviation
            of the gaussian distribution.
        proportion_pcic (float): proportion of data caused by parallel clock induced charge
        proportion_scic (float): proportion of data caused by serial clock induced charge   
        exp_loc (float): the location of the start of the exponential
            distribution and the SCiC distribution.
    """

    def __init__(self, beta=10.0, mu=10.0, sigma=1.0, proportion_pcic=0.3, 
                 proportion_scic=0.5, register_count=100, exp_loc=5, **kwargs):
        self.beta: float = kwargs.get('beta', beta)
        self.mu: float = kwargs.get('mu', mu)
        self.sigma: float = kwargs.get('sigma', sigma)
        self.proportion_pcic: float = kwargs.get('proportion_scic', proportion_pcic)
        self.proportion_scic: float = kwargs.get('proportion_pcic', proportion_scic)
        self.exp_loc: float = kwargs.get('exp_loc', exp_loc)
        self.register_count: int = kwargs.get('register_count', register_count)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'beta: {self.beta:.5f} | mu: {self.mu:.5f} | ' \
               f'sigma: {self.sigma:.5f} | exp_loc: {self.exp_loc:.5f} \
                | proportion pcic: {self.proportion_pcic:.5f} | proportion scic: {self.proportion_scic:.5f} '

    def as_list(self) -> list:
        """Gets the parameters as a list.

        Returns:
            beta, mu, sigma, and proportion as a list
        """
        return [self.beta, self.mu, self.sigma, self.proportion_pcic, self.proportion_scic, self.exp_loc]

    def max_parameter_difference(self, other) -> float:
        """Get the largest difference in parameters to another GaussianExponentialParameters.

        Compares this object to another GaussianExponentialParameters object parameter by
        parameter and returns the absolute value of the largest difference.

        Args:
            other (GaussianExponentialParameters): the parameters to compare to. This operation
                is symmetric.

        Returns:
            The largest pairwise difference in the parameter list.
        """
        return max([abs(i[0] - i[1]) for i in zip(self.as_list(), other.as_list())])


class EmccdModelMixture:

    """Fits a mixture of a Gaussian and Exponential and Scic distribution to data in a Numpy array

    This implementation uses Expectation Maximization -- referred to as EM in these docs --
    to iteratively converge on solutions for four unknown parameters:

        - mu: the mean of the Gaussian/Normal distribution, the bias
        - sigma: the standard deviation of the Gaussian/Normal distribution, the readout noise
        - beta: the mean of the Exponential distribution, or the gain of the serial register
        - proportion_pcic: proportion of data caused by parallel clock induced charge
        - proporion_scic: proportion of data caused by serial clock induced charge

    TODO: Link to Appendix with derivations of update conditions.

    Args:
        data (np.numarray): single dimensional array of data to fit distributions to
        exp_loc (float): location of the exponential distribution
        max_iterations (int): terminate after this number of EM steps
        convergence_tolerance (float): terminate if no parameter moves by more than this value
        distribution_fix (bool): support use case where gaussian mu and exponential offset are locked
    """

    def __init__(self,
                 data: np.numarray,
                 max_iterations=100,
                 convergence_tolerance=0.001,
                 distribution_fix=False,
                 **kwargs):

        self.convergence_tolerance: float = convergence_tolerance
        self.data: np.numarray = data
        self.parameters = EmccdModelParameters( **kwargs)
        self.parameters_updated = EmccdModelParameters( **kwargs)
        self.max_iterations: int = max_iterations
        self.distribution_fix: bool = distribution_fix
        self.pcic = stats.expon(loc=self.parameters.exp_loc, scale=self.parameters.beta)
        self.norm = stats.norm(loc=self.parameters.mu, scale=self.parameters.sigma)
        self.scic = SerialCicDistribution(exp_loc=self.parameters.exp_loc, 
                                          register_count=self.parameters.register_count, 
                                          register_gain=self.parameters.beta)
        print('Starting parameters.....')    
        print(self.parameters_updated)

    def _apply_and_sum(self, func: callable) -> float:
        """Applies a function to the data and returns the sum of the array.

        Args:
            func (callable): a callable with the signature func(val: float) -> float.

        Returns:
            The sum of the data vector after applying func.
        """
        return np.sum(np.vectorize(func)(self.data))

    def _expectation_is_scic(self, val: float) -> float:
        """Computes (prob_scic)/(prob_gaussian + prob_scic + prob_pcic) for the value passed
           with some protection against underflow.
        """
        gaussian_density = self.norm.logpdf(val)
        pcic_density = self.pcic.logpdf(val)
        scic_density = self.scic.logpdf(val)

        log_prob_gaussian = gaussian_density + np.log(1-self.parameters.proportion_scic
                                                      -self.parameters.proportion_pcic)
        log_prob_pcic = pcic_density + np.log(self.parameters.proportion_pcic)
        log_prob_scic = pcic_density + np.log(self.parameters.proportion_scic)

        expectation_is_scic = np.exp(
                log_prob_scic - np.log(np.exp(log_prob_gaussian)+
                 np.exp(log_prob_scic)+np.exp(log_prob_pcic))
        )
        if expectation_is_scic == np.nan:
            return 0
        else:
            return expectation_is_scic


    def _expectation_is_pcic(self, val: float) -> float:
        """Computes (prob_pcic)/(prob_gaussian + prob_scic + prob_pcic) for the value passed
           with some protection against underflow.
        """
        gaussian_density = self.norm.logpdf(val)
        pcic_density = self.pcic.logpdf(val)
        scic_density = self.scic.logpdf(val)

        log_prob_gaussian = gaussian_density + np.log(1-self.parameters.proportion_scic
                                                      -self.parameters.proportion_pcic)
        log_prob_pcic = pcic_density + np.log(self.parameters.proportion_pcic)
        log_prob_scic = pcic_density + np.log(self.parameters.proportion_scic)

        expectation_is_pcic = np.exp(
                log_prob_pcic - np.log(np.exp(log_prob_gaussian)+np.exp(log_prob_scic)+ 
                np.exp(log_prob_pcic))
        )
        if expectation_is_pcic == np.nan:
            return 0
        else:
            return expectation_is_pcic


    def _update_beta(self) -> None:
        """Updates the beta parameter (mean/scale) of the exponential distribution.
        """
        self.parameters_updated.beta = \
            self._apply_and_sum(lambda x: (self._expectation_is_pcic(x)) * (x-self.parameters_updated.exp_loc)) / \
            self._apply_and_sum(lambda x: (self._expectation_is_pcic(x)))

    def _update_mu(self) -> None:
        """Updates the mu parameter (mean/location) of the gaussian distribution.
        """
        self.parameters_updated.mu = \
            self._apply_and_sum(lambda x: (1-self._expectation_is_pcic(x)-self._expectation_is_scic(x)) * x) / \
            self._apply_and_sum(lambda x: (1-self._expectation_is_pcic(x)-self._expectation_is_scic(x)))

    def _update_exp_loc(self) -> None:
        """Updates the location parameter of the exponential distribution.

         Note:
            Assumes this parameter is fixed unless it track the Gausian Mu.  There might be a update
            equation for the normal case that could be added in future
        """
        if self.distribution_fix is True:
           self.parameters_updated.exp_loc = self.parameters_updated.mu
           
    def _update_sigma(self) -> None:
        """Updates the sigma parameter (standard deviation/scale) of the gaussian distribution.

        Note:
            Updating the standard deviation of the normal distribution requires the updated
            mean for this iteration to be in updated_parameters for behavior to be defined.
        """
        sigma_squared = \
            self._apply_and_sum(lambda x: ((1-self._expectation_is_pcic(x)-self._expectation_is_scic(x))) * (x - self.parameters_updated.mu) ** 2) / \
            self._apply_and_sum(lambda x: ((1-self._expectation_is_pcic(x)-self._expectation_is_scic(x))))
        self.parameters_updated.sigma = math.sqrt(sigma_squared)

    def _update_proportion_pcic(self) -> None:
        """Updates the proportion of the data that is likelier gaussian.
        """
        pcic_total = self._apply_and_sum(lambda x: np.nan_to_num(self.pcic.logpdf(x)) >
                                                       np.nan_to_num(self.scic.logpdf(x)) and
                                                      np.nan_to_num(self.pcic.logpdf(x)) >
                                                       np.nan_to_num(self.norm.logpdf(x)) )
        self.parameters_updated.proportion_pcic = pcic_total / len(self.data)

    def _update_proportion_scic(self) -> None:
        """Updates the proportion of the data that is likelier gaussian.
        """
        scic_total = self._apply_and_sum(lambda x: np.nan_to_num(self.scic.logpdf(x)) >
                                                       np.nan_to_num(self.pcic.logpdf(x)) and
                                                      np.nan_to_num(self.scic.logpdf(x)) >
                                                       np.nan_to_num(self.norm.logpdf(x)) )
        self.parameters_updated.proportion_scic = scic_total / len(self.data)

    def _sync_parameters(self) -> None:
        """Copies parameters_updated into parameters.

        This prepares the state of GaussianExponentialMixture for another iteration
        of the EM algorithm with the parameters updated from the previous iteration.
        """
        self.parameters = deepcopy(self.parameters_updated)

    def _update_pdfs(self) -> None:
        """Updates PDFs of normal and exponential with new parameters.

        Since the parameters are stored separately from the PDFs for now, updates
        need to be applied on each iteration.
        """
        self.norm = stats.norm(loc=self.parameters_updated.mu, scale=self.parameters_updated.sigma)
        self.pcic = stats.expon(loc=self.parameters_updated.exp_loc, scale=self.parameters_updated.beta)
        self.scic = SerialCicDistribution(exp_loc=self.parameters_updated.exp_loc, \
                                          register_count=self.parameters.register_count, \
                                          register_gain=self.parameters_updated.beta)

    def _check_parameter_differences(self) -> float:
        """Compares the newly updated parameters to the previous iteration.

        Returns:
            This returns the largest pairwise difference between parameter values for
            use in determining the convergence of EM.
        """
        return self.parameters.max_parameter_difference(self.parameters_updated)

    def em_step(self) -> None:
        """Performs one EM step on the data and stores the result in updated_parameters.

        Note:
            While This method can be used safely independently, it is advisable to use `self.fit`
            in almost all cases outside of debugging since it handles a iteration and
            tracks convergence.
        """
        self._sync_parameters()
        self._update_beta()
        self._update_mu()
        self._update_exp_loc()
        self._update_sigma()
        self._update_pdfs()
        self._update_proportion_pcic()
        self._update_proportion_scic()

    def fit(self) -> None:
        """Performs EM steps until convergence criteria are satisfied.

        Note:
            If your data is large or your convergence criteria is strict this may take
            a long time.

            To debug, consider running `em_step` directly and monitoring parameter movement
            and iteration time.
        """
        self.em_step()
        iters = 1
        while iters < self.max_iterations and self._check_parameter_differences() > self.convergence_tolerance:
            self.em_step()
            iters += 1
            print(self.parameters_updated)
        self._sync_parameters()

    def logpdf(self, val)-> float:
        """Evaluates the density of the logpdf of the EmccdModelMixture.
        """
        weighted_log_gaussian_density = np.log(1-self.parameters.proportion_scic
                                               -self.parameters.proportion_pcic) + self.norm.logpdf(val)
        weighted_log_scic_density = np.log((self.parameters.proportion_scic)) + self.scic.logpdf(val)
        weighted_log_pcic_density = np.log((self.parameters.proportion_pcic)) + self.pcic.logpdf(val)
        log_density = np.log(np.exp(weighted_log_gaussian_density)+np.exp(weighted_log_pcic_density)+np.exp(
                                   weighted_log_scic_density))
        return log_density

    def pdf(self, val) -> float:
        """Evaluates the density of the pdf of the GaussianExponentialMixture.
        """
        return np.exp(self.logpdf(val))
