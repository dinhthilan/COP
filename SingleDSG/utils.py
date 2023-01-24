# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 15:37:30 2020

@author: Thi Lan Dinh
"""
import numpy as np
import math
import time
import matplotlib.pyplot as plt

def max_func(vec_val,mat_grad):
    ind_max=np.argmax(vec_val)
    return vec_val[ind_max][0],mat_grad[:,ind_max]

def abs_func(val,grad):
    return abs(val), np.sign(val)*grad

def abs_h_eq(x,h_eq):
    vec_val_eq,mat_grad_eq = h_eq(x)
    for i in range(len(vec_val_eq)):
        vec_val_eq[i],mat_grad_eq[:,i] = abs_func(vec_val_eq[i],mat_grad_eq[:,i])
    return vec_val_eq,mat_grad_eq

def max_cons(x,f_ineq,h_eq):
    vec_val_ineq,mat_grad_ineq = f_ineq(x)
    vec_val_eq,mat_grad_eq = abs_h_eq(x,h_eq)
    vec_val = np.concatenate((vec_val_ineq,vec_val_eq),axis=0)
    mat_grad = np.concatenate((mat_grad_ineq,mat_grad_eq),axis = 1)
    return max_func(vec_val,mat_grad)



def update(x_hat,s_hat,x0,lambd0,x,lambd,beta,s_x,s_lambd,f_ineq,h_eq,obj):
    val_obj,grad_obj = obj(x)
    val_max_cons,grad_max_cons = max_cons(x,f_ineq,h_eq)
    G_x = grad_obj+lambd*grad_max_cons
    G_lambd = val_max_cons
    norm_G = math.sqrt(sum(G_x**2)+G_lambd**2)
    s_x += G_x/norm_G
    s_lambd -=  G_lambd/norm_G
    
    s_hat += 1/norm_G
    x_hat += x/norm_G
    
    x = x0-s_x/beta
    lambd = lambd0 - s_lambd/beta
    beta += 1/beta
    
    return x_hat,s_hat,x,lambd,beta,s_x,s_lambd

def plot(epoch,obj,check = 'Infeas'):
    
    fig,ax=plt.subplots()
    ax.plot(range(1,epoch),obj,label='MPDS')
    ax.set_xscale('log',base=2)
    plt.xlabel('Iterations')
    if check == 'Infeas':
        plt.ylabel('Feasibility')
    else:
        plt.ylabel('Value')
    
    legend = ax.legend(loc='upper right')   
    plt.show()

def run(x_hat,s_hat,x0,lambd0,beta,s_x,s_lambd,f_ineq,h_eq,obj,Print=False,history=False,numiter=200):
   
    x,lambd=x0,lambd0
    FEAS = []
    VAL = []

    n=len(x0)
    
    if Print:
        print('SingleDSG: version 1.0')
        print('Number of variables: n = {} '.format(n))
    
    t = time.time()
    i=0
    for k in range(1,numiter):

        
        x_hat,s_hat,x,lambd,beta,s_x,s_lambd = update(x_hat,s_hat,x0,lambd0,x,lambd,beta,s_x,s_lambd,f_ineq,h_eq,obj)
        
        x_bar = 1/s_hat*x_hat

        Val = obj(x_bar)[0]
        Infeas = abs(max_cons(x_bar,f_ineq,h_eq)[0])
        
        if history:
               
                FEAS.append(Infeas)
                VAL.append(Val)
    
        
        if k==2**i or k==numiter-1:
        
            if Print:
                print('Iter = {},  Val = {:.4f},  Infeas = {:.4f}'.format(k,Val,Infeas))
            i+=1
    if Print:
        print('Total time: {:.4f} '.format(time.time()-t))
        
    return Val,x_bar,VAL,FEAS


   