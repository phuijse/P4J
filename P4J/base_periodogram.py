import numpy as np

class BasePeriodogram:
    
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
        
    def frequency_grid_evaluation(self, fmin=0.0, fmax=1.0, fresolution=1e-4):
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
        
        """
        self.freq_step_coarse = fresolution
        freqs = np.arange(start=np.amax([fmin, fresolution]), stop=fmax, 
                          step=fresolution, dtype=np.float32)
        self.per, self.per_single_band = self._compute_periodogram(freqs)
        self.freq = freqs 
    
    def find_local_maxima(self, n_local_optima=10):
        
        local_optima_index = 1+np.where((self.per[1:-1] > self.per[:-2]) & (self.per[1:-1] > self.per[2:]))[0]
        
        if(len(local_optima_index) < n_local_optima):
            print("Warning: Not enough local maxima found in the periodogram")
            return
        # Keep only n_local_optima
        best_local_optima = local_optima_index[np.argsort(self.per[local_optima_index])][::-1]
        if n_local_optima > 0:
            best_local_optima = best_local_optima[:n_local_optima]
        else:
            best_local_optima = best_local_optima[0]
            
        return best_local_optima
    
    def finetune_best_frequencies(self, fresolution=1e-5, n_local_optima=10):
        """
        Computes the selected criterion over a grid of frequencies 
        around a specified amount of  local optima of the periodograms. This
        function is intended for additional fine tuning of the results obtained
        with grid_search
        """
        #assert fresolution < self.freq_step_coarse, "Frequency step error: fine tune step is greater than or equal to coarse step"
        local_optima_index = self.find_local_maxima(n_local_optima)
        
        for local_optimum_index in local_optima_index:
            fmin = self.freq[local_optimum_index] - self.freq_step_coarse
            fmax = self.freq[local_optimum_index] + self.freq_step_coarse
            freqs_fine = np.arange(fmin, fmax, step=fresolution, dtype=np.float32)
            pers_fine = self._compute_periodogram(freqs_fine)
            self._update_periodogram(local_optimum_index, freqs_fine, pers_fine)

        # Sort them in descending order
        idx = np.argsort(self.per[local_optima_index])[::-1]
        if n_local_optima > 0:
            self.best_local_optima = local_optima_index[idx]
        else:
            self.best_local_optima = local_optima_index