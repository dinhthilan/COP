# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:15:54 2020

@author: Thi Lan Dinh
"""

import func as func
import numpy as np
import utils
import Plot

l = 4
n = 3
m = 1
A = np.random.rand(l,n)
z = np.array([0,0,0]).reshape(n,1)
b = np.matmul(A,z)
x0 = np.random.rand(n,1).reshape(n,)
lambd0 = 0
x_hat = np.zeros([n,])
s_hat = 0
s_x = np.zeros([n,])
s_lambd = 0
beta = 1
numiter=int(1e+3)


def h_eq(x): 
    vec_val = np.matmul(A,x.reshape(n,1))-b
    mat_grad = A.transpose()
    return vec_val.reshape(l,1),mat_grad

Val,x_bar,VAL,FEAS = utils.run(x_hat,
                     s_hat,
                     x0,
                     lambd0,beta,s_x,s_lambd,
                     func.eval_f_ineq,h_eq,func.eval_f0,
                     Print=True,history=True,numiter=numiter)


Plot.plot(range(len(FEAS)),[FEAS],Labels=['SingleDSG'],check='feas')
Plot.plot(range(len(FEAS)),[VAL],Labels=['SingleDSG'],check='Val')











