import numpy as np
import scipy.stats as sts
from typing import Tuple


class MonteCarloOptionPricing:
    def __init__(self, r, S0: float, K: float, T: float, sigma: float, div_yield: float = 0.0,
                 simulation_rounds: int = 10000, no_of_slices: int = 4, fix_random_seed: bool or int = False):
        """
        An important reminder, by default the implementation assumes constant interest rate and volatility.
        To allow for stochastic interest rate and vol, run Vasicek/CIR for stochastic interest rate and
        run Heston for stochastic volatility.
        :param S0: current price of the underlying asset (e.g. stock)
        :param K: exercise price
        :param T: time to maturity, in years, can be float
        :param r: interest rate, by default we assume constant interest rate model
        :param sigma: volatility (in standard deviation) of the asset annual returns
        :param div_yield: annual dividend yield
        :param simulation_rounds: in general, monte carlo option pricing requires many simulations
        :param no_of_slices: between time 0 and time T, the number of slices PER YEAR, e.g. 252 if trading days are required
        :param fix_random_seed: boolean or integer
        """
        assert sigma >= 0, 'volatility cannot be less than zero'
        assert S0 >= 0, 'initial stock price cannot be less than zero'
        assert T >= 0, 'time to maturity cannot be less than zero'
        assert div_yield >= 0, 'dividend yield cannot be less than zero'
        assert no_of_slices >= 0, 'no of slices per year cannot be less than zero'
        assert simulation_rounds >= 0, 'simulation rounds cannot be less than zero'
        
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.div_yield = float(div_yield)

        self.no_of_slices = int(no_of_slices)
        self.simulation_rounds = int(simulation_rounds)

        self._dt = self.T / self.no_of_slices

        self.mue = r  # under risk-neutral measure, asset expected return = risk-free rate
        self.r = np.full((self.simulation_rounds, self.no_of_slices), r * self._dt)
        self.discount_table = np.exp(np.cumsum(-self.r, axis=1))

        self.sigma = np.full((self.simulation_rounds, self.no_of_slices), sigma)

        self.terminal_prices = []

        self.z_t = np.random.standard_normal((self.simulation_rounds, self.no_of_slices))

        if type(fix_random_seed) is bool:
            if fix_random_seed:
                np.random.seed(15000)
        elif type(fix_random_seed) is int:
            np.random.seed(fix_random_seed)

    def heston(self, kappa: float, theta: float, sigma_v: float, rho: float = 0.0) -> np.ndarray:
        """
        When asset volatility (variance NOT sigma!) follows a stochastic process.
        Heston stochastic volatility with Euler discretisation. Notice the native Euler discretisation could lead to
        negative volatility. To mitigate this issue, several methods could be used. Here we choose the full truncation
        method.
        dv(t) = kappa[theta - v(t)] * dt + sigma_v * sqrt(v(t)) * dZ
        :param: kappa: rate at which vt reverts to theta
        :param: theta: long-term variance
        :param: sigma_v: sigma of the volatility
        :param: rho: correlation between the volatility and the rate of return
        :return: stochastic volatility array
        """
        _variance_v = sigma_v ** 2
        assert 2 * kappa * theta > _variance_v  # Feller condition

        # step 1: simulate correlated zt
        _mu = np.array([0, 0])
        _cov = np.array([[1, rho], [rho, 1]])
        _zt = sts.multivariate_normal.rvs(mean=_mu, cov=_cov, size=(self.simulation_rounds, self.no_of_slices))
        _variance_array = np.full((self.simulation_rounds, self.no_of_slices),
                                  self.sigma[0, 0] ** 2)
        self.z_t = _zt[:, :, 0]
        _zt_v = _zt[:, :, 1]

        # step 2: simulation
        for i in range(1, self.no_of_slices):
            _previous_slice_variance = np.maximum(_variance_array[:, i - 1], 0)
            _drift = kappa * (theta - _previous_slice_variance) * self._dt
            _diffusion = sigma_v * np.sqrt(_previous_slice_variance) * \
                         _zt_v[:, i - 1] * np.sqrt(self._dt)
            _delta_vt = _drift + _diffusion
            _variance_array[:, i] = _variance_array[:, i - 1] + _delta_vt

        # re-define the interest rate and volatility path
        self.sigma = np.sqrt(np.maximum(_variance_array, 0))
        return self.sigma

    def stock_price_simulation(self) -> Tuple[np.ndarray, float]:
        self.exp_mean = (self.mue - self.div_yield - (self.sigma ** 2.0) * 0.5) * self._dt
        self.exp_diffusion = self.sigma * np.sqrt(self._dt)

        self.price_array = np.zeros((self.simulation_rounds, self.no_of_slices))
        self.price_array[:, 0] = self.S0

        for i in range(1, self.no_of_slices):
            self.price_array[:, i] = self.price_array[:, i - 1] * np.exp(
                self.exp_mean[:, i - 1] + self.exp_diffusion[:, i - 1] * self.z_t[:, i - 1]
            )

        self.terminal_prices = self.price_array[:, -1]
        self.stock_price_expectation = np.mean(self.terminal_prices)
        self.stock_price_standard_error = np.std(self.terminal_prices) / np.sqrt(len(self.terminal_prices))

        '''print('-' * 64)
        print(
            " Number of simulations %4.1i \n S0 %4.1f \n K %2.1f \n Maximum Stock price %4.2f \n"
            " Minimum Stock price %4.2f \n Average stock price %4.3f \n Standard Error %4.5f " % (
                self.simulation_rounds, self.S0, self.K, np.max(self.terminal_prices),
                np.min(self.terminal_prices), self.stock_price_expectation, self.stock_price_standard_error
            )
        )
        print('-' * 64)'''

        return self.stock_price_expectation, self.stock_price_standard_error

    def european_call(self) -> Tuple[float, float]:
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'

        self.terminal_profit = np.maximum((self.terminal_prices - self.K), 0.0)

        self.expectation = np.mean(self.terminal_profit * np.exp(-np.sum(self.r, axis=1)))
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        '''print('-' * 64)
        print(
            " European call monte carlo \n S0 %4.1f \n K %2.1f \n"
            " Call Option Value %4.3f \n Standard Error %4.5f " % (
                self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)'''
        return self.expectation

    def european_put(self, empirical_call: float or None = None) -> float:
        """
        Use put call parity (incl. continuous dividend) to calculate the put option value
        :param empirical_call: can be calculated or observed call option value
        :return: put option value
        """

        if empirical_call is not None:
            self.european_call_value = self.european_call()
        else:
            self.european_call_value = empirical_call

        self.put_value = self.european_call_value + np.exp(-np.sum(self.r, axis=1)) * self.K - np.exp(
            -self.div_yield * self.T) * self.S0
        
        return self.put_value[0]