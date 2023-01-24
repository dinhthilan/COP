# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:00:19 2020

@author: Thi Lan Dinh
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time
__all__ = ['normsq', 'plot', 'abs_func','abs_h_eq','max_func','max_cons','update','run']


def normsq(vec):
    return sum(vec**2)

def plot(epoch,obj,check = 'feas'):
    
    fig,ax=plt.subplots()
    ax.plot(range(1,epoch),obj,label='MPDS')
    plt.xlabel('Iterations')
    if check == 'feas':
        plt.ylabel('Feasibility')
    else:
        plt.ylabel('Value')
    legend = ax.legend(loc='upper right')   
            
    plt.show()
 
def abs_func(val,grad):
    return abs(val), np.sign(val)*grad

def abs_h_eq(x,h_eq):
    vec_val_eq,mat_grad_eq = h_eq(x)
    for i in range(len(vec_val_eq)):
        vec_val_eq[i],mat_grad_eq[:,i] = abs_func(vec_val_eq[i],mat_grad_eq[:,i])
    return vec_val_eq,mat_grad_eq

def max_func(vec_val,mat_grad):
    ind_max=np.argmax(vec_val)
    return vec_val[ind_max][0],mat_grad[:,ind_max]

def max_cons(x,f_ineq,h_eq):
    vec_val_ineq,mat_grad_ineq = f_ineq(x)
    vec_val_eq,mat_grad_eq = abs_h_eq(x,h_eq)
    vec_val = np.concatenate((vec_val_ineq,vec_val_eq),axis=0)
    mat_grad = np.concatenate((mat_grad_ineq,mat_grad_eq),axis = 1)
    return max_func(vec_val,mat_grad)

def update(n,x,x_star,val_star,eval_f0,eval_f_ineq,h_eq,eps=0.1):
    val,sub = eval_f0(x)
    val_ineq,sub_ineq = max_cons(x,eval_f_ineq,h_eq)
    val_star,sub_star = eval_f0(x_star)
    delta=1e-5
    if val_ineq <= eps:
        x = x - eps/(normsq(sub)+delta)*sub
        if val < val_star:
            val_star,x_star = val,x
    else:
        x = x - val_ineq/(normsq(sub_ineq)+delta)*sub_ineq
    return x,x_star,val,val_star,max(0,val_ineq)

def run(m,x,eval_f0,eval_f_ineq,h_eq,eps=0.1,numiter=200,Print=False,history=False):
    n=len(x)
    Feas_INEQ = []
    VAL = []
    if Print:
        print('SG: version 1.0')
        print('Number of variables: n = {} '.format(n))
        print('Number of enequality constrains: m = {}'.format(m))
    i=0
    t = time.time()
    x_star = x
    val_star = eval_f0(x_star)[0]
    for k in range(1,numiter):
        x,x_star,val,val_star,feas_ineq = update(n,x,x_star,val_star,eval_f0,eval_f_ineq,h_eq,eps=eps)
        if Print and (k==2**i or k==numiter-1):
            print('Iter = {},  Val = {:.4f},  Feas_ineq = {:.4f}'.format(k,val,feas_ineq))
            i+=1
        if history:
            Feas_INEQ.append(feas_ineq)
            VAL.append(val)
    if Print:
        print('Total time: {:.4f} '.format(time.time()-t))
    return val_star,x_star,val,x,Feas_INEQ,VAL