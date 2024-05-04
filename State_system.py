## moudule of metapopulation epidemic travel bubble problem
## this moudule will relaize two ideas, 1. 

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
import time


class State_solution:
    
    def __init__(self, N_p, N_0, sigma, X, kpa, Z, delta, beta, params):
        self.N_p = N_p     # population size
        self.N_0 = N_0     # initial pandemic data
        self.sigma = sigma # commuting rate
        self.X = X         # transportation matrix (dict not matrix)
        self.kpa = kpa     # regional daily contact rate
        #self.hos_cap = hos_cap
        self.Z = Z         # country and region dict
        self.delta = delta # transportation control variable
        self.beta = beta
        self.p_t = params['p_t']  # percentage allowed to travel
        self.v = params['v']      # infection period
        self.xi = params['xi']    # latent period
        self.tau = params['tau']  # return rate of daily commuters
        self.eta = params['eta']      # coefficient to change the daily contact
        self.t_s = params['t_s']  # period of stay
        self.gamma_s = params['gamma_s']
        self.gamma_v = params['gamma_v']
        self.c_v = params['c_v'] # {j:c_j^V}
        self.c_i = params['c_i'] # {j:c_j^I}
        self.c_b = params['c_b'] # {j:c_j^beta}
        self.c_d = params['c_d'] # {j:c^delta} #cost of transportation control
        self.t_f = params['t_f'] # ratio of inbound tourists beside return residents
        self.T = params['T']     # research period
        self.init()
        
        
        
    def init(self):
        self.alpha = {j:0.5 for j in self.N_p}
        
        self.kpa_hat = {}
        for j in self.kpa:
            X_j = sum(self.X[l] for l in  self.X if l[1]==j)
            self.kpa_hat[j] = [self.kpa[j]+(self.eta-1)*self.alpha[j]*self.t_s*self.delta[t]*X_j*self.kpa[j]/self.N_p[j] for t in range(len(self.delta))]
        #print(self.kpa_hat)
        self.sigma_sum = {j: sum(self.sigma[l] for l in self.sigma if l[0]==j) for j in self.N_p}

    
    
    def Lam(self, j, N, t):#  j is subpopulation area code, N is the population state at time t
        N_star = self.N_p[j]
        #kpa_j = self.kpa_hat[j][t]
        
        lam_jj_s = self.kpa_hat[j][t]*self.gamma_s/self.N_p[j] * (N[j][3]/(1+self.sigma_sum[j]/self.tau) + sum(N[i][3]*self.sigma[(i,j)]/(self.tau+self.sigma_sum[i])  for i in self.N_p if i!= j))/(1 + self.sigma_sum[j]/self.tau)
        
        lam_jj_v = self.kpa_hat[j][t]*self.gamma_v/self.N_p[j] * (N[j][3]/(1+self.sigma_sum[j]/self.tau) + sum(N[i][3]*self.sigma[(i,j)]/(self.tau+self.sigma_sum[i])  for i in self.N_p if i!= j))/(1 + self.sigma_sum[j]/self.tau) 
        
        lam_ji_s = sum( self.sigma[(j,i)]* self.kpa_hat[i][t]* self.gamma_s/self.N_p[i]* (N[i][3]/(1+self.sigma_sum[i]/self.tau) + sum(N[l][3]*self.sigma[(l,i)]/(self.tau+self.sigma_sum[l])  for l in self.N_p if l!= i))  for i in self.N_p if i != j )/(self.tau + self.sigma_sum[j])
        
        lam_ji_v = sum( self.sigma[(j,i)]* self.kpa_hat[i][t]* self.gamma_v/self.N_p[i]* (N[i][3]/(1+self.sigma_sum[i]/self.tau) + sum(N[l][3]*self.sigma[(l,i)]/(self.tau+self.sigma_sum[l])  for l in self.N_p if l!= i)) for i in self.N_p if i != j )/(self.tau + self.sigma_sum[j])
        
        lam_j_s = lam_jj_s + lam_ji_s
        lam_j_v = lam_jj_v + lam_ji_v
        return lam_j_s, lam_j_v
    
    
    
    
    def get_P(self,N, i): # calculate the air travel probability of different state in different areas
        Int = self.N_p[i]
        P_n = [ N[i][0]/Int, N[i][1]/Int, N[i][2]/Int, N[i][3]/Int, N[i][4]/Int ]
        return P_n
    
    
    
    def SD(self, j, N, t):# the state dynamic calculation
        lam_s, lam_v = self.Lam(j, N, t)
        beta_j = self.beta[j][t]
        Delta_j = np.array([[-lam_s-beta_j,0, 0, 0, 0],\
                            [beta_j, -lam_v, 0, 0, 0],\
                            [lam_s, lam_v, -self.xi, 0, 0,],\
                            [0, 0, self.xi, -self.v, 0],\
                            [0, 0, 0, self.v, 0]])
        n_j = np.array(N[j])
        n_j_new = Delta_j@n_j + n_j
        X_t = self.X
        P = {i: self.get_P(N, i) for i in self.N_0}
        X_ij_s = sum(X_t[(i,j)]*P[i][0] for i in self.N_0 if i != j)*self.delta[t]
        X_ij_v = sum(X_t[(i,j)]*P[i][1] for i in self.N_0 if i != j)*self.delta[t]
        X_ij_l = sum(X_t[(i,j)]*P[i][2] for i in self.N_0 if i != j)*self.delta[t]
        X_ij_i = sum(X_t[(i,j)]*P[i][3] for i in self.N_0 if i != j)*self.delta[t]
        X_ij_r = sum(X_t[(i,j)]*P[i][4] for i in self.N_0 if i != j)*self.delta[t]
        X_ji_s = sum(X_t[(j,i)]*P[j][0] for i in self.N_0 if i != j)*self.delta[t]
        X_ji_v = sum(X_t[(j,i)]*P[j][1] for i in self.N_0 if i != j)*self.delta[t]
        X_ji_l = sum(X_t[(j,i)]*P[j][2] for i in self.N_0 if i != j)*self.delta[t]
        X_ji_i = sum(X_t[(j,i)]*P[j][3] for i in self.N_0 if i != j)*self.delta[t]
        X_ji_r = sum(X_t[(j,i)]*P[j][4] for i in self.N_0 if i != j)*self.delta[t]
        
        N_j_tp1 = n_j_new + np.array([X_ij_s-X_ji_s,X_ij_v-X_ji_v, X_ij_l-X_ji_l, X_ij_i-X_ji_i,  X_ij_r-X_ji_r])
        return N_j_tp1.tolist(), lam_s, lam_v
    
    
    
    
    def get_res(self):
        M = {0: self.N_0}
        L_s = {j:[] for j in self.N_p}
        L_v = {j:[] for j in self.N_p}
        #T = int(len(date)/2)
        for t in range(1,self.T):
            M[t]={}
            for j in self.N_0:
                M[t][j], lam_js, lam_jv = self.SD(j, M[t-1], t)
                L_s[j].append(lam_js)
                L_v[j].append(lam_jv)
        R_res = {}
        for j in self.N_0:
            R_res[j]={}
            R_res[j][0]=[]
            R_res[j][1]=[]
            R_res[j][2]=[]
            R_res[j][3]=[]
            R_res[j][4]=[]
            for t in M:
                R_res[j][0].append(M[t][j][0])
                R_res[j][1].append(M[t][j][1])
                R_res[j][2].append(M[t][j][2])
                R_res[j][3].append(M[t][j][3])
                R_res[j][4].append(M[t][j][4])
        
        cost = 0
        for t in range(self.T):
            for j in self.N_0:
                cost += (self.c_i[j]*1000/self.N_p[j]*(R_res[j][3][t])**2 + self.c_v*R_res[j][0][t]*self.beta[j][t] + self.c_b[j]*self.beta[j][t]**2)
            cost += self.c_d*(1-self.delta[t])**2 
        
        return R_res, cost, L_s, L_v
    
    
    