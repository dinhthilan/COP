# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:35:24 2020

@author: Thi Lan Dinh
"""


import Wong2 as w2

import numpy as np
import Plot 
import sys
sys.path.append('/home/hoanganh/Desktop/math-topics/COPFC/codes/COPFC')

import SG.utils as SG
import SingleDSG.utils as SingleDSG
import MultiDSG.utils as MultiDSG
import PDS.utils as PDS
import math


import Plot 
import warnings
warnings.filterwarnings("ignore")

n = 10
m = 3
x0 = np.array([2,3,5,5,1,2,7,3,6,10]).reshape(10,1)
#x0 = np.random.rand(n,1)
l = 0
A = np.zeros([l,n])
b = np.zeros([l,1])
lambd = np.zeros([m,1])
v = np.random.rand(l,1)
numiter = int(1e+5)
val_exact = 24.306209
delta=0.5
eps=1e-3

print('numiter=',numiter)
print('delta=',delta)
print('eps=',eps)
print('val_exact = ', val_exact)

def obj_new(x):
    val,grad=w2.obj(x.reshape(n,1))
    return val,grad.reshape(n,)

def cons_new(x):
    vec_val,mat_grad=w2.cons(x.reshape(n,1))
    return vec_val,mat_grad.reshape(n,m)

def h_eq(x): 
    vec_val = np.matmul(A,x.reshape(n,1))-b
    mat_grad = A.transpose()
    return vec_val.reshape(l,1),mat_grad


#############################################


_,_,_,_,FEAS1,VAL1 = SG.run(m,x0.reshape(n,),obj_new,cons_new,h_eq,eps=eps,
                 numiter=numiter,Print=True,history=True)

print('gap={:.2f}\%'.format(PDS.gap(val_exact, VAL1[-1])*100))

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
print('gap={:.2f}\%'.format(PDS.gap(val_exact, VAL2[-1])*100))
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
print('gap={:.2f}\%'.format(PDS.gap(val_exact, VAL3[-1])*100))
####################################
x0=x0.reshape(n,1)


s=1
print('s = ',s)
_,_,_,_,FEAS4,VAL4 = PDS.run(A,b,x0,v,lambd,m,w2.obj,w2.cons,
                 Print=True,history=True,delta=delta,rho=1/s,s =s,numiter=numiter,
                 eps=eps)
print('gap={:.2f}\%'.format(PDS.gap(val_exact, VAL4[-1])*100))

####################################
s=1.5
print('s = ',s)
_,_,_,_,FEAS5,VAL5 = PDS.run(A,b,x0,v,lambd,m,w2.obj,w2.cons,
                 Print=True,history=True,delta=delta,rho=1/s,s =s,numiter=numiter,
                 eps=eps)

print('gap={:.2f}\%'.format(PDS.gap(val_exact, VAL5[-1])*100))
####################################
s=2
print('s = ',s)
_,_,_,_,FEAS6,VAL6 = PDS.run(A,b,x0,v,lambd,m,w2.obj,w2.cons,
                 Print=True,history=True,delta=delta,rho=1/s,s =s,numiter=numiter,
                 eps=eps)
print('gap={:.2f}\%'.format(PDS.gap(val_exact, VAL6[-1])*100))

####################################

Plot.plot(range(1,len(FEAS1)+1),[FEAS1,FEAS2,FEAS3,FEAS4,FEAS5,FEAS6],Labels=['SG','SingleDSG','MultiDSG','PDS with s=1','PDS with s=1.5','PDS with s=2'],check='feas')
Plot.plot(range(1,len(FEAS1)+1),[VAL1,VAL2,VAL3,VAL4,VAL5,VAL6],Labels=['SG','SingleDSG','MultiDSG','PDS with s=1','PDS with s=1.5','PDS with s=2'],check='Val',loc='lower left')
