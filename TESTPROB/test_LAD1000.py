# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:08:59 2020

@author: Thi Lan Dinh
"""
import numpy as np
import Rand_Prob as rp
import SetPythonPath
import SG.utils as SG
import SingleDSG.utils as SingleDSG
import MultiDSG.utils as MultiDSG
import PDS.utils as PDS
import math

import Plot 
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


n_bar = 1000

m_bar = math.ceil(n_bar*2)
n = n_bar + m_bar
D = rp.rand_mat(m_bar,n_bar)
w = rp.rand_mat(m_bar,1)

m=1
l = 2*n_bar
def obj(z):# z = (x,y)
    subgrad = np.zeros([n,1])
    subgrad[n_bar:] = np.sign(z[n_bar:])
    val =  np.linalg.norm(z[n_bar:],ord = 1)   
    return val,subgrad 

def cons(z):
    return np.zeros([m,1]),np.zeros([n,m])


A = np.concatenate((D,-np.identity(m_bar)),axis = 1)
l=A.shape[0]
b=np.zeros([l,1])

lambd = np.zeros([m,1])
v = np.random.rand(m_bar,1)
numiter= int(1e+5)
delta=0.99
x0 = np.random.rand(n,1)
print('delta = ',delta)
print('numiter = ',numiter)
x0=x0.reshape(n,)
lambd0 = 0
x_hat = np.zeros([n,])
s_hat = 0
s_x = np.zeros([n,])
s_lambd = 0
beta = 1

eps=1e-3
print('eps=',eps)

def obj_new(x):
    val,grad=obj(x.reshape(n,1))
    return val,grad.reshape(n,)

def cons_new(x):
    vec_val,mat_grad=cons(x.reshape(n,1))
    return vec_val,mat_grad.reshape(n,m)

def h_eq(x): 
    vec_val = np.matmul(A,x.reshape(n,1))-b
    mat_grad = A.transpose()
    return vec_val.reshape(l,1),mat_grad


#############################################


_,_,_,_,FEAS1,VAL1 = SG.run(m,x0.reshape(n,),obj_new,cons_new,h_eq,eps=eps,
                 numiter=numiter,Print=True,history=True)

#########################################
#DSG

x0=x0.reshape(n,)
lambd0 = 0
x_hat = np.zeros([n,])
s_hat = 0
s_x = np.zeros([n,])
s_lambd = 0
beta = 1


_,_,VAL2,FEAS2 = SingleDSG.run(x_hat,
                     s_hat,
                     x0,lambd0,beta,s_x,s_lambd,
                     cons_new,h_eq,obj_new,
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


_,_,VAL3,FEAS3 = MultiDSG.run(n,m,l,x_hat,s_hat,
                     x0,mu0,theta0,
                     beta,
                     s_x,s_mu,s_theta,
                     cons_new,h_eq,obj_new,
                     rho=0.5,
                     Print=True,history=True,numiter=numiter)

####################################
x0=x0.reshape(n,1)


s=1
print('s = ',s)
_,_,_,_,FEAS4,VAL4 = PDS.run(A,b,x0,v,lambd,m,obj,cons,
                 Print=True,history=True,delta=delta,rho=1/s,s =s,numiter=numiter,
                 eps=eps)


####################################
s=1.5
print('s = ',s)
_,_,_,_,FEAS5,VAL5 = PDS.run(A,b,x0,v,lambd,m,obj,cons,
                 Print=True,history=True,delta=delta,rho=1/s,s =s,numiter=numiter,
                 eps=eps)


####################################
s=2
print('s = ',s)
_,_,_,_,FEAS6,VAL6 = PDS.run(A,b,x0,v,lambd,m,obj,cons,
                 Print=True,history=True,delta=delta,rho=1/s,s =s,numiter=numiter,
                 eps=eps)


####################################

Plot.plot(range(1,len(FEAS1)+1),[FEAS1,FEAS2,FEAS3,FEAS4,FEAS5,FEAS6],Labels=['SG','SingleDSG','MultiDSG','PDS with s=1','PDS with s=1.5','PDS with s=2'],check='feas')
Plot.plot(range(1,len(FEAS1)+1),[VAL1,VAL2,VAL3,VAL4,VAL5,VAL6],Labels=['SG','SingleDSG','MultiDSG','PDS with s=1','PDS with s=1.5','PDS with s=2'],check='Val',loc='lower left')
