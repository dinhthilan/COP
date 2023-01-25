# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 18:46:47 2020

@author: Thi Lan Dinh
"""
import numpy as np
import Plot 
import sys
sys.path.append('./')

import SG.utils as SG
import SingleDSG.utils as SingleDSG
import MultiDSG.utils as MultiDSG
import PDS.utils as PDS


n_bar = 2
N1 = 100*n_bar
N2 = N1
N = N1 + N2
Z1 = np.random.rand(N1,n_bar)
Z2 = -np.random.rand(N2,n_bar)
Y1 = np.ones([N1,1])
Y2 = -Y1
Y = np.concatenate((Y1,Y2),axis = 0)
n = n_bar+N+1
m = 1
def funcl(i,x,yi):
    val = max(0,1-yi*x[n_bar+i])
    subgrad = np.zeros([n,1])
    if val >0:
        subgrad[n_bar+i] = -yi
    return val/N,subgrad.reshape(n,)/N

def sum_func(vec_val,mat_grad):
    return sum(vec_val),mat_grad.sum(axis = 1).reshape(n,1)

def norm2(x):
    val =  1/2*sum(x[:n_bar-1]**2)
    subgrad = np.zeros([n,1])
    subgrad[:n_bar-1] = x[:n_bar-1].reshape(n_bar-1,1)
    return val,subgrad.reshape(n,)

def obj(x):
    vec_val = np.zeros([N+1,1]).reshape(N+1,)
    mat_grad = np.zeros([n,N+1])
    
    for i in range(N):
        vec_val[i],mat_grad[:,i] = funcl(i,x,Y[i])
    vec_val[N],mat_grad[:,N] = norm2(x)
    return sum_func(vec_val,mat_grad)

def cons(x):
    return np.zeros([m,1]),np.zeros([n,m])

Z = np.concatenate((Z1,Z2),axis = 0)
A = np.concatenate((Z,-np.identity(N),-np.ones([N,1])),axis = 1)
l = N
b = np.zeros([l,1])
lambd = np.zeros([m,1])
v = np.random.rand(l,1)
numiter= int(1e+4)
delta=0.99
x0 = np.random.rand(n,1)
print('delta = ',delta)
print('numiter = ',numiter)
eps=1e-3
print('eps=',eps)


def obj_new(x):
    val,grad=obj(x)
    return val,grad.reshape(n,)

def cons_new(x):
    vec_val,mat_grad=cons(x)
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
Plot.plot(range(1,len(FEAS1)+1),[VAL1,VAL2,VAL3,VAL4,VAL5,VAL6],Labels=['SG','SingleDSG','MultiDSG','PDS with s=1','PDS with s=1.5','PDS with s=2'],check='Val',loc='upper right')
