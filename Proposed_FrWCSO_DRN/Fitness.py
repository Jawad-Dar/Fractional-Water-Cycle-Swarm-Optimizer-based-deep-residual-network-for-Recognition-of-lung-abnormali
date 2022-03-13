import hoggorm as ho
import numpy as np
from numpy import dot
from numpy.linalg import norm

def func(soln,xx,yy): # fitness function
    def lee_dist(a, b): # Lee Distance
        m = 4
        x = sum(a - b) # x
        y = m - x # y
        lee = min(abs(x), abs(y)) # finding lee distance
        return lee

    xx_ = (np.transpose(xx)) # transposing data to select the column values
    LD = 0 # initializing lee distance
    for i in range(len(xx_)):
        if (soln[i] == 1): # find the lee distance only for the solution containing 1
            LD += lee_dist(xx_[i], np.array(yy)) # lee distance
    return LD


def fit_func(soln,xx,yy): # fitness function
    def RV_Coeff(a, b):
        a = [a]
        b = [b]
        r_coeff = ho.RVcoeff([a, b])# RV Coefficient
        return r_coeff[0][1]

    def Cosine_Similarity(a,b):  # calculate Cosine Similarity
        cos_sim = dot(a, b) / (norm(a) * norm(b))
        return cos_sim

    xx_ = (np.transpose(xx)) # transposing data to select the column values
    F = 0 # initializing Fit
    for i in range(len(xx_)):
        if (soln[i] == 1): # find the RV Coefficient and Cosine similarity only for the solution containing 1
            rv = RV_Coeff(xx_[i], np.array(yy))
            cs = Cosine_Similarity(xx_[i], np.array(yy))
            f = (rv+cs)/2
            F += f
    return F
