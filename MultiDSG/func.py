# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:12:00 2020

@author: Thi Lan Dinh
"""
import numpy as np

def eval_f0(variant):
    subgrad = np.sign(variant)
    val =  np.linalg.norm(variant,ord = 1)   
    return val,subgrad.reshape(3,)


def eval_funcx1(x):
    return x[0],np.array([1,0,0]).reshape(3,1)

def eval_funczero(x):
    return 0,np.zeros([3,1])

def eval_max(func1,func2,x):
    vala,subgrada=func1(x)
    valb,subgradb=func2(x)
    val,subgrad=vala,subgrada
    if val<valb:
        val,subgrad=valb,subgradb
    return val,subgrad

def eval_funcx2(x):
    return x[1],np.array([0,1,0]).reshape(3,1)



def eval_funcx3(x):
    return x[2],np.array([0,0,1]).reshape(3,1)



def eval_sum(func1,func2,x):
    vala,subgrada=func1(x)
    valb,subgradb=func2(x)
    return vala+valb,subgrada+subgradb
   
                     
def eval_f_ineq(x):
    def func1(x):
        return eval_max(eval_funcx1,eval_funczero,x)
    def func2(x):
        return eval_max(eval_funcx2,eval_funcx3,x)
    val,grad=eval_sum(func1,func2,x)
    return val.reshape(1,1),grad












    
