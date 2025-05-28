from scipy.stats import norm

class LIB_acq_fun:
    def __init__(self,mu, std,acq_type,xi=0,kappa=0):
        self.mu = mu
        self.std = std
        self.acq_type = acq_type
        self.xi = xi
        self.kappa = kappa

    def EI(self,y_best):
        a = y_best - self.mu-self.xi
        std = self.std + 1E-9
        z = a/(std)
        ac = a*norm.cdf(z) + std*norm.pdf(z)
        return ac


    def LCB(self):
        ac = -(self.mu - self.kappa*self.std)
        return ac

    def POI(self,y_best):
        a = y_best-self.mu-self.xi
        std = self.std + 1E-9
        z = a/std
        ac = norm.cdf(z)
        return ac

    def greedy(self):
        ac = -self.mu
        return ac

    def acq_fun(self,y_best):
        if self.acq_type == 'EI':
            ac = self.EI(y_best)
        elif self.acq_type == 'LCB':
            ac = self.LCB()
        elif self.acq_type == 'POI':
            ac = self.POI(y_best)
        elif self.acq_type == 'greedy':
            ac = self.greedy()
        return ac
