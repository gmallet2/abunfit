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
        """Prior uniforme sur R⁺ⁿ (tous les coefficients ≥ 0)."""
        if np.any(params < 0):
            return -np.inf
        return 0.0

    @staticmethod
    def log_likelihood(params   : np.ndarray,
                       modmatrix: np.ndarray,
                       data     : np.ndarray) -> float:
        """
        ln L = −½ χ²
        data shape : (n_elements, 2) — [valeur, erreur]
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
    def run(modmatrix     : np.ndarray,
            data          : np.ndarray,
            p0_best       : np.ndarray,
            n_walkers     : int   = 64,
            n_steps       : int   = 3000,
            n_burn        : int   = 500,
            perturbation  : float = 1e-3,
            progress      : bool  = True) -> tuple[np.ndarray, object]:
        """
        Lance le MCMC avec emcee.

        Paramètres :
            modmatrix    : (n_models, n_elements)
            data         : (n_elements, 2)
            p0_best      : point de départ (résultat du fit moindres carrés)
            n_walkers    : nombre de walkers (doit être ≥ 2 × n_params)
            n_steps      : nombre de pas total par walker
            n_burn       : nombre de pas de burn-in à ignorer
            perturbation : amplitude de la perturbation initiale autour de p0_best
            progress     : afficher la barre de progression emcee

        Retourne :
            flat_samples : (n_walkers × (n_steps − n_burn), n_params)
            sampler      : objet emcee.EnsembleSampler (pour diagnostics)
        """

        n_params = len(p0_best)
        n_walkers = max(n_walkers, 2 * n_params + 2)

        # Position initiale : légère perturbation autour du best-fit
        # On s'assure que tous les walkers démarrent avec des valeurs positives
        rng = np.random.default_rng(seed=42)
        p0  = p0_best[np.newaxis, :] * (
            1.0 + perturbation * rng.standard_normal((n_walkers, n_params))
        )
        p0  = np.abs(p0)   # garantir la positivité au départ

        sampler = emcee.EnsembleSampler(
            n_walkers, n_params,
            MCMC.log_posterior,
            args=(modmatrix, data),
        )
        sampler.run_mcmc(p0, n_steps, progress=progress)

        flat_samples = sampler.get_chain(discard=n_burn, flat=True)
        return flat_samples, sampler