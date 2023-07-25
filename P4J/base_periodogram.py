import numpy as np
import abc
import logging
from typing import List


class BasePeriodogram(abc.ABC):
    def get_best_frequency(self, fid=None):
        if fid is None:
            best_idx = np.argmax(self.per)
        else:
            best_idx = np.argmax(self.per_single_band[fid])
        return self.freq[best_idx]
        
    def get_best_frequencies(self):
        """
        Returns the best n_local_max frequencies
        """
        
        return self.freq[self.best_local_optima], self.per[self.best_local_optima]
        
    def get_periodogram(self, fid=None):
        if fid is None:
            return self.freq, self.per
        else:
            return self.freq, self.per_single_band[fid]
        
    def frequency_grid_evaluation(
            self, fmin=0.0, fmax=1.0, fresolution=1e-4, log_period_spacing=False):
        """ 
        Computes the selected criterion over a grid of frequencies 
        with limits and resolution specified by the inputs. After that
        the best local maxima are evaluated over a finer frequency grid
        
        Parameters
        ---------
        fmin: float
            starting frequency
        fmax: float
            stopping frequency
        fresolution: float
            step size in the frequency grid
        log_period_spacing: bool
            If True uses a log-spaced period grid.
            If False (default) uses a linear-spaced frequency grid.
        """
        self.freq_step_coarse = fresolution

        # TODO: remove. log_period_spacing is not right.
        if log_period_spacing:
            period_min = np.log10(1.0/fmax)
            period_max = np.log10(1.0/fmin)
            grid_len = int((fmax - fmin) / fresolution)
            periods = np.logspace(period_min, period_max, grid_len)
            freqs = 1.0 / periods
            freqs = freqs[::-1]
        else:
            freqs = np.arange(start=np.amax([fmin, fresolution]), stop=fmax,
                              step=fresolution, dtype=np.float32)
        self.per, self.per_single_band = self._compute_periodogram(freqs)
        self.freq = freqs

    def optimal_frequency_grid_evaluation(
            self, smallest_period: float, largest_period: float, shift: float = 0.1):
        lc_time_length = self.get_lc_time_length()
        smallest_frequency = 1.0/largest_period
        largest_frequency = 1.0/smallest_period
        f_range = largest_frequency - smallest_frequency
        grid_size = int(np.ceil(f_range * lc_time_length / (2.0 * shift)))
        grid_size = max(grid_size, 1_000)
        self.freq_step_coarse = f_range / (grid_size - 1)

        frequency_grid = np.linspace(
            smallest_frequency,
            largest_frequency,
            grid_size,
            dtype=np.float32
        )
        self.per, self.per_single_band = self._compute_periodogram(frequency_grid)
        self.freq = frequency_grid
    
    def find_local_maxima(self, n_local_optima=10) -> List[int]:
        local_optima_index = 1+np.where(
            (self.per[1:-1] > self.per[:-2]) & (self.per[1:-1] > self.per[2:]))[0]
        
        n_optima_found = len(local_optima_index)
        if n_optima_found < n_local_optima:
            logging.warning(f"Only {n_optima_found} local maxima were found in the periodogram")
            n_local_optima = min(n_local_optima, n_optima_found)

        # Keep only n_local_optima
        best_local_optima = local_optima_index[np.argsort(self.per[local_optima_index])][::-1]
        best_local_optima = best_local_optima[:n_local_optima]
            
        return best_local_optima
    
    def finetune_best_frequencies(self, fresolution=1e-5, n_local_optima=10):
        """
        Computes the selected criterion over a grid of frequencies 
        around a specified amount of  local optima of the periodograms. This
        function is intended for additional fine-tuning of the results obtained
        with grid_search
        """
        if self.freq_step_coarse < fresolution:
            logging.warning(
                'Frequency step error: fine tune step is '
                'greater than or equal to coarse step')
        local_optima_index = self.find_local_maxima(n_local_optima)
        
        for local_optimum_index in local_optima_index:
            if local_optimum_index - 1 < 0:
                freq_before = self.freq[0] - (self.freq[1] - self.freq[0])
            else:
                freq_before = self.freq[local_optimum_index - 1]

            if local_optimum_index + 1 >= len(self.freq):
                freq_after = self.freq[-1] + (self.freq[-1] - self.freq[-2])
            else:
                freq_after = self.freq[local_optimum_index + 1]

            fine_grid = np.linspace(
                freq_before,
                freq_after,
                2 * int(np.ceil(self.freq_step_coarse / fresolution)))

            pers_fine = self._compute_periodogram(fine_grid)
            self._update_periodogram(local_optimum_index, fine_grid, pers_fine)

        # Sort them in descending order
        idx = np.argsort(self.per[local_optima_index])[::-1]
        if n_local_optima > 0:
            self.best_local_optima = local_optima_index[idx]
        else:
            self.best_local_optima = local_optima_index

    def optimal_finetune_best_frequencies(self, times_finer: float = 10.0, n_local_optima: int = 10):
        """
        Computes the selected criterion over a grid of frequencies
        around a specified amount of  local optima of the periodograms. This
        function is intended for additional fine-tuning of the results obtained
        with grid_search
        """

        fresolution = self.freq_step_coarse / times_finer
        self.finetune_best_frequencies(fresolution, n_local_optima=n_local_optima)

    @abc.abstractmethod
    def _update_periodogram(self, local_optimum_index, freqs_fine, pers_fine):
        pass

    @abc.abstractmethod
    def _compute_periodogram(self, freqs_fine):
        pass
