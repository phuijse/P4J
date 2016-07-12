import numpy as np
from .regression import find_beta_WMCC, find_beta_OLS, find_beta_WLS
from .dictionary import harmonic_dictionary

class periodogram:
    def __init__(self, method='WMCC', M=1):
        self.method = method
        self.M = M
        
    def fit(self, t, y, dy):
        self.t = t
        self.y = y
        self.dy = dy
        self.T = t[-1] - t[0]
        
    def grid_search(self, fmin=0.0, fmax=1.0, fres_coarse=1.0, fres_fine=0.1, n_local_max=10):
        # Perform a grid search using a coarse frequency step
        freq = np.arange(np.amax([fmin, fres_coarse/self.T]), fmax, step=fres_coarse/self.T)
        Nf = len(freq)
        per = np.zeros(shape=(Nf,))
        for k in range(0, Nf):
            Phi = harmonic_dictionary(self.t, freq[k], self.M)
            if self.method == 'WMCC':
                beta, cost_history, _ = find_beta_WMCC(self.y, Phi, self.dy)
                per[k] = cost_history[-1]
        # Find the local minima and do analysis with finer frequency step
        local_max_index = []
        for k in range(1, Nf-1):
            if per[k-1] < per[k] and per[k+1] < per[k]:
                local_max_index.append(k)
        local_max_index = np.array(local_max_index)
        best_local_max = local_max_index[np.argsort(per[local_max_index])][::-1]
        # Do finetuning
        for j in range(0, n_local_max):
            freq_fine = freq[best_local_max[j]] - fres_coarse/self.T
            for k in range(0, int(2.0*fres_coarse/fres_fine)):
                Phi = harmonic_dictionary(self.t, freq_fine, self.M)
                _, cost_history, _ = find_beta_WMCC(self.y, Phi, self.dy)
                if cost_history[-1] > per[best_local_max[j]]:
                    per[best_local_max[j]] = cost_history[-1]
                    freq[best_local_max[j]] = freq_fine
                freq_fine += fres_fine/self.T
        
        return freq, per
