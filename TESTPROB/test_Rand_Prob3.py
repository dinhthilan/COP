# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:43:18 2020

@author: Thi Lan Dinh
"""

import Rand_Prob as rp

import numpy as np
import Plot 
import sys
sys.path.append('./')
from DSG.schemes.DSG import utils as utils_DSG
from MultiDSG import utils as utils_MDSG
import PDS.utils as utils_PDS
import math


n = 10
l = math.ceil(n/7)
m = 2
A = rp.rand_mat(l,n)
c = rp.rand_mat(n,1).reshape(n,)
x0 = np.random.rand(n,1)
x_bar = rp.rand_mat(n,1)
x_bar[0] = 0
x_bar[1] = 0
x_bar[2] = 1

#x_bar*=np.random.rand(1)[0]
b = np.matmul(A,x_bar)

lambd = np.zeros([m,1])
v = np.random.rand(l,1)
numiter= int(1e+5)
def obj(x):
    return rp.linear_obj(c,x,n)
def constr3(x):
    return rp.cons3(n,x)

s=1.5
print('s = ',s)
val_star,x_star,val,x,FEAS1,VAL1 = utils_PDS.run(A,b,x0,v,lambd,m,obj,constr3,
                 Print=True,history=True,delta=0.5,rho=1/s,s =s,numiter=numiter,
                 eps=0.01)
print('Val = ',val)
print('-'*20)


#########################################
#DSG

x0=x0.reshape(n,)
lambd0 = 0
x_hat = np.zeros([n,])
s_hat = 0
s_x = np.zeros([n,])
s_lambd = 0
beta = 1

def obj_new(x):
    val,grad=rp.linear_obj(c,x,n)
    return val,grad.reshape(n,)

def h_eq(x): 
    vec_val = np.matmul(A,x.reshape(n,1))-b
    mat_grad = A.transpose()
    return vec_val.reshape(l,1),mat_grad

Val,x_bar,VAL2,FEAS2 = utils_DSG.run(x_hat,
                     s_hat,
                     x0,
                     lambd0,beta,s_x,s_lambd,
                     constr3,h_eq,obj_new,
                     Print=True,history=True,numiter=numiter)

########################################
mu0 = np.zeros([m,])
theta0=np.zeros([l,])

x_hat = np.zeros([n,])
s_hat = 0

s_x = np.zeros([n,])
s_mu = np.zeros([m,])
s_theta = np.zeros([l,])

beta = 1


Val,x_bar,VAL3,FEAS3 = utils_MDSG.run(n,m,l,x_hat,s_hat,
                     x0,mu0,theta0,
                     beta,
                     s_x,s_mu,s_theta,
                     constr3,h_eq,obj_new,
                     rho=0.5,
                     Print=True,history=True,numiter=numiter)

Plot.plot(range(1,len(FEAS1)+1),[FEAS1,FEAS2,FEAS3],Labels=['MPDS','DSG','MDSG'],check='feas')
Plot.plot(range(1,len(FEAS1)+1),[VAL1,VAL2,VAL3],Labels=['MPDS','DSG','MDSG'],check='Val')





