"""
Implements MCMC for abunfit application
"""
import numpy as np
import emcee


class MCMC:
    """
    Exploring (with bayesian implementation) the parameters space, using emcee

    Stat. model :
        Prior is uniform
        Likelihood : gaussian χ² 
    posterior is prop to exp(−χ²/2) 
    """

    @staticmethod
    def log_prior(params: np.ndarray) -> float:
        """
        Uniform prior for params over 0.
        Inputs :
            - params (np.array) : array of parameters ;
        output :
            0 or -infty ;
        """
        if np.any(params < 0):
            return -np.inf
        return 0.0

    @staticmethod
    def log_likelihood(params   : np.ndarray,
                       modmatrix: np.ndarray,
                       data     : np.ndarray) -> float:
        """
        ln L = -1/2 chi_2
        data shape : (n_elements, 2) — [value, error]
        """
        model    = np.dot(params, modmatrix)
        errors   = np.where(data[:, 1] > 0, data[:, 1], 1.0)
        residuals = (data[:, 0] - model) / errors
        return -0.5 * np.sum(residuals ** 2)

    @staticmethod
    def log_posterior(params   : np.ndarray,
                      modmatrix: np.ndarray,
                      data     : np.ndarray) -> float:
        lp = MCMC.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + MCMC.log_likelihood(params, modmatrix, data)

    @staticmethod
    def run(modmatrix : np.ndarray,data : np.ndarray, p0_best : np.ndarray, n_walkers = 64,
            n_steps = 3000, n_burn = 500, perturbation = 1e-3, progress = True) :
        """
        Launch MCMC with mcee
        Inputs :
            modmatrix  : (n_models, n_elements)
            data      : (n_elements, 2)
            p0_best      : beginning point (result of the fit typically)

            n_walkers   (int) : number of emcee walkers (has to be > 2*n_models)
            n_steps      : number of steps per walker
            n_burn       : burn-in (first steps to ignore)
            perturbation : initial dispersion qround the best-fit
            progress     : do we want a progress bar
        Outputs :
            flat_samples : (n_walkers * (n_steps * n_burn), n_params)
            sampler      : (objet emcee.EnsembleSampler) (for diagnosis)

        """
        n_params = len(p0_best)
        n_walkers = max(n_walkers, 2 * n_params + 2)
        rng = np.random.default_rng(seed=42)
        p0  = p0_best[np.newaxis, :] * (
            1.0 + perturbation * rng.standard_normal((n_walkers, n_params))
        )
        p0  = np.abs(p0) 

        sampler = emcee.EnsembleSampler(
            n_walkers, n_params,
            MCMC.log_posterior,
            args=(modmatrix, data),
        )
        sampler.run_mcmc(p0, n_steps, progress=progress)

        flat_samples = sampler.get_chain(discard=n_burn, flat=True)
        return flat_samples, sampler