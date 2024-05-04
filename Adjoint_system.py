## moudule of metapopulation epidemic travel bubble problem
## this moudule will relaize two ideas, 1. 
### reverse the onjective function to obtain maximum

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
import time


class Adjoint_solution:
    
    def __init__(self, L_s, L_v, X, kpa, N_p,sigma, N, beta_max,beta_min, del_min, beta, delta, params ):
        self.L_s = L_s     # lambda_S
        self.L_v = L_v     # Lambda_V
        self.X = X         # transportation matrix (dict not matrix)
        self.kpa = kpa     # regional daily contact rate
        self.N_p = N_p     # population size
        self.sigma = sigma
        self.N = N         # pandemic data
        self.b_max = beta_max    # upper limit of vaccination rate
        self.b_min = beta_min
        self.d_min = del_min
        self.T = params['T']
        self.c_i = params['c_i']     # cost coefficient of infection
        self.c_v = params['c_v']     # cost coefficient of vaccination rate
        self.xi = params['xi']
        self.tau = params['tau']  # return rate of daily commuters
        self.v = params['v']
        self.eta = params['eta']      # coefficient to change the daily contact
        self.t_s = params['t_s']  # period of stay
        self.gamma_s = params['gamma_s']
        self.gamma_v = params['gamma_v']
        self.c_b = params['c_b']
        self.c_d = params['c_d']
        self.t_f = params['t_f'] # ratio of inbound tourists beside return residents
        self.beta = beta
        self.delta = delta
        self.init()
        
    def init(self):
        self.X_bar = {j: sum(self.X[l] for l in self.X if l[1]==j) for j in self.N_p}
        self.alpha = {i : 0.5 for i in self.N_p}
        self.kpa_hat = {}
        for j in self.kpa:
            X_j = sum(self.X[l] for l in  self.X if l[1]==j)
            self.kpa_hat[j] = [self.kpa[j]+(self.eta-1)*self.alpha[j]*self.t_s*self.delta[t]*X_j*self.kpa[j]/self.N_p[j] for t in range(len(self.delta))]
        self.sigma_sum = {j: sum(self.sigma[l] for l in self.sigma if l[0]==j) for j in self.N_p}
        
        
    def get_mu(self):
        
        #### calculating the derivative of lambda to I_j
        lam_ds = {}
        lam_dv = {}
        for j in self.kpa_hat:
            lam_ds[j] = [self.kpa_hat[j][t]*self.gamma_s/self.N_p[j]*(self.tau/(self.tau+self.sigma_sum[j]))**2 + sum(self.kpa_hat[i][t]*self.gamma_s/self.N_p[i]*(self.sigma[(j,i)]/(self.tau+self.sigma_sum[j]))**2 for i in self.N_p if i != j) for t in range(len(self.kpa_hat[j]))]
            lam_dv[j] = [self.kpa_hat[j][t]*self.gamma_v/self.N_p[j]*(self.tau/(self.tau+self.sigma_sum[j]))**2 + sum(self.kpa_hat[i][t]*self.gamma_v/self.N_p[i]*(self.sigma[(j,i)]/(self.tau+self.sigma_sum[j]))**2 for i in self.N_p if i != j) for t in range(len(self.kpa_hat[j]))]
        
        ##### obtain mu in backward sweep
        mu_s = {j:[0] for j in self.N_p}       # mu_s 
        mu_v = {j:[0] for j in self.N_p}       # mu_v  
        mu_l = {j:[0] for j in self.N_p}       # mu_l 
        mu_i = {j:[0] for j in self.N_p}       # mu_i 
        mu_r = {j:[0] for j in self.N_p}       # mu_r 
        for t in range(1,self.T):
            for j in self.N_p:
                mu_s_g = (mu_s[j][0] -mu_l[j][0])*self.L_s[j][-t]  +(mu_s[j][0]- mu_v[j][0]-self.c_v)*self.beta[j][-t] -self.delta[-t]/self.N_p[j]*sum((mu_s[i][0]- mu_s[j][0])*self.X[(j,i)] for i in self.N_p if i != j)
                mu_s[j].insert(0, mu_s[j][0]-mu_s_g)
                
                mu_v_g = (mu_v[j][0] -mu_l[j][0])*self.L_v[j][-t]  -self.delta[-t]/self.N_p[j]*sum((mu_v[i][0]- mu_v[j][0])*self.X[(j,i)] for i in self.N_p if i != j)
                mu_v[j].insert(0, mu_v[j][0]-mu_v_g)
                
                mu_l_g = (mu_l[j][0] - mu_i[j][0])*self.xi   -self.delta[-t]/self.N_p[j]*sum((mu_l[i][0]- mu_l[j][0])*self.X[(j,i)] for i in self.N_p if i != j)
                mu_l[j].insert(0, mu_l[j][0]-mu_l_g)
                
                mu_i_g = (mu_i[j][0]-mu_r[j][0])*self.v   -2*self.c_i[j]*self.N[j][3][-t]/(self.N_p[j]/1000)  + (mu_s[j][0] - mu_l[j][0])*self.N[j][0][-t]*lam_ds[j][-t]  + (mu_v[j][0] - mu_l[j][0])*self.N[j][1][-t]*lam_dv[j][-t]  -self.delta[-t]/self.N_p[j]*sum((mu_i[i][0]- mu_i[j][0])*self.X[(j,i)] for i in self.N_p if i != j)
                mu_i[j].insert(0, mu_i[j][0]-mu_i_g)
                
                mu_r_g =  - self.delta[-t]/self.N_p[j]*sum((mu_r[i][0] -mu_r[j][0])*self.X[(j,i)] for i in self.N_p if i != j)
                mu_r[j].insert(0, mu_r[j][0]-mu_r_g)
        #print(mu_s[1], mu_v[1], mu_l[1], mu_i[1], mu_r[1])
        return mu_s, mu_v, mu_l, mu_i, mu_r

    
    
    def Lam_d(self, j, t):#  j is subpopulation area code, N is the population state at time t
        #N_star = self.N_p[j]
        #kpa_j = self.kpa[j]
        #lam_jj_3 = 0
        #sigma_j = {j:sum(self.sigma[l] for l in self.sigma if l[0]==j) for j in self.N_p}
        #lam_del = 0
        
        lam_jj_del = (self.eta-1)*self.t_s*self.alpha[j]*self.X_bar[j]*self.kpa[j]/(self.N_p[j]**2)* (self.N[j][3][t]*self.tau/(self.tau+self.sigma_sum[j]) + sum(self.N[i][3][t]*self.sigma[(i,j)]/(self.tau+self.sigma_sum[i]) for i in self.N_p if i != j))/(1+self.sigma_sum[j]/self.tau)
        
        lam_ji_del = sum( self.alpha[i] *self.X_bar[i]*self.kpa[i]/(self.N_p[i]**2) * (self.N[i][3][t]*self.tau/(self.tau+self.sigma_sum[i]) + sum(self.N[l][3][t]*self.sigma[(l,i)]/(self.tau+self.sigma_sum[l]) for l in self.N_p if l != i))*self.sigma[(j,i)] for i in self.N_p if i != j)*(self.eta-1)*self.t_s/(self.tau+self.sigma_sum[j])
             
        return lam_jj_del+lam_ji_del
    
    
    
    def get_beta(self):
        beta_star = {j:[] for j in self.N_p}
        
        mu_s, mu_v, mu_l, mu_i, mu_r = self.get_mu()
        for j in beta_star:
            for t in range(self.T):
                b_star = self.N[j][0][t]*( mu_s[j][t]- self.c_v - mu_v[j][t])/(2*self.c_b[j]) 
                
                beta_star[j].append(min(max(self.b_min, b_star), self.b_max[t]))
        
        return  beta_star
    
    
    
    def get_delta(self):
        mu_s, mu_v, mu_l, mu_i, mu_r = self.get_mu()
        delta_star = []
        
        for t in range(self.T):
            lam_j = {j: self.Lam_d(j, t) for j in self.N_p}
            lam_sj = {j: lam_j[j]*self.gamma_s for j in self.N_p}
            lam_vj = {j: lam_j[j]*self.gamma_v for j in self.N_p}
            
            del_ = sum( (mu_l[j][t]- mu_s[j][t])* lam_sj[j]*self.N[j][0][t]  + (mu_l[j][t]- mu_v[j][t])* lam_vj[j]*self.N[j][1][t]        + mu_s[j][t]*sum(self.X[(i,j)]*self.N[i][0][t]/self.N_p[i] - self.X[(j,i)]*self.N[j][0][t]/self.N_p[j] for i in self.N_p if i != j)       + mu_v[j][t]*sum(self.X[(i,j)]*self.N[i][1][t]/self.N_p[i] - self.X[(j,i)]*self.N[j][1][t]/self.N_p[j] for i in self.N_p if i != j)       + mu_l[j][t]*sum(self.X[(i,j)]*self.N[i][2][t]/self.N_p[i] - self.X[(j,i)]*self.N[j][2][t]/self.N_p[j] for i in self.N_p if i != j)       + mu_i[j][t]*sum(self.X[(i,j)]*self.N[i][3][t]/self.N_p[i] - self.X[(j,i)]*self.N[j][3][t]/self.N_p[j] for i in self.N_p if i != j)       + mu_r[j][t]*sum(self.X[(i,j)]*self.N[i][4][t]/self.N_p[i] - self.X[(j,i)]*self.N[j][4][t]/self.N_p[j] for i in self.N_p if i != j)  for j in self.N_p)
            
            _del = 1-del_/self.c_d/2
            delta_star.append(min(max(self.d_min, _del), 1))
        
        return  delta_star
        
    
        
    