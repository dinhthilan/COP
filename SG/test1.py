# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:15:54 2020

@author: Thi Lan Dinh
"""

import numpy as np
import func as func
import  utils
import Plot


l = 4
n = 3
m = 1
A = np.random.rand(l,n)
z = np.array([0,0,0]).reshape(n,1)
b = np.matmul(A,z)
x0 = np.random.rand(n,1).reshape(n,)
numiter=int(1e+3)


def h_eq(x): 
    vec_val = np.matmul(A,x.reshape(n,1))-b
    mat_grad = A.transpose()
    return vec_val.reshape(l,1),mat_grad

val_star,x_star,val,x,FEAS,VAL = utils.run(m,x0,func.eval_f0,func.eval_f_ineq,h_eq,eps=0.001,
                 numiter=numiter,Print=True,history=True)

Plot.plot(range(len(FEAS)),[FEAS],Labels=['SG'],check='feas')
Plot.plot(range(len(FEAS)),[VAL],Labels=['SG'],check='Val')
