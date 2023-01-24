# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:15:54 2020

@author: Thi Lan Dinh
"""

import sys
import Plot
import func as func
import numpy as np

import utils

l = 4
n = 3
m = 1
A = np.random.rand(l,n)
z = np.array([0,0,0]).reshape(n,1)
b = np.matmul(A,z)
x0 = np.random.rand(n,1)
lambd = np.random.rand(m,1)
v = np.random.rand(l,1)
Numiter=int(1e+3)


val,x,FEAS2,VAL2 = utils.run(A,b,x0,v,lambd,m,func.eval_f0,func.eval_f_ineq,
                 Print=True,history=True,delta=1,rho=1,s = 1,numiter=Numiter,eps=0.001)
print('-'*20)



#Plot.plot(range(len(FEAS2)),[FEAS2],Labels=['PDS with s=1.5'],check='feas')
#Plot.plot(range(len(FEAS2)),[VAL2],Labels=['PDS with s=1.5'],check='Val')

