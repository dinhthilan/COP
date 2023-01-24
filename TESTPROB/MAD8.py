# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:24:06 2020

@author: Thi Lan Dinh
"""
import numpy as np

def f1(x):
    n=20
    val=-1+x[0]**2+sum(x[1:])
    grad=np.ones([n,1])
    grad[0] = 2*x[0]
    return val[0],grad


def fi(x,c,k):
    n = 20
    val = -1 + c*x[k-1]**2 + sum(np.delete(x,k-1))
    grad = np.ones([n,1])
    grad[k-1] = 2*c*x[k-1]
    return val[0],grad
def f38(n,x):
    val = -1+x[19]**2 + sum(np.delete(x,19))
    grad = np.ones([n,1])
    grad[19] = 2*x[19]
    return val[0],grad.reshape(20,)
    
def max_with_zero(val,grad):
    if val > 0:
        grad_new = grad
        val_new = val
    else:
        grad_new = np.zeros([grad.shape[0],1])
        val_new = 0
    return val_new, grad_new.reshape(20,)

def abs_func(val,grad):
    return abs(val), np.sign(val)*grad.reshape(20,)

def sum_func(vec_val,mat_grad):
    return sum(vec_val),mat_grad.sum(axis = 1)
def max_func(n,vec_val,mat_grad):
    indx = np.argmax(vec_val)
    val = vec_val[indx]
    grad = mat_grad[:,indx]
    return val[0],grad.reshape(n,1)
def obj(x):
    n = 20
    m = 38
    vec_val = np.zeros([m,1])
    mat_grad = np.zeros([n,m])
    val,grad = f1(x)
    vec_val[0],mat_grad[:,0] = abs_func(val,grad)
    
    for i in range(2,38):
        if i%2==0:
            c = 1
            k = int((i+2)/2)
        else:
            c = 2
            k = int((i+1)/2)
            
        val,grad = fi(x,c,k)
        vec_val[i-1],mat_grad[:,i-1] = abs_func(val,grad)
    vec_val[37],mat_grad[:,37]= f38(n,x)
    return max_func(n,vec_val,mat_grad)
    

def cons(x):
    n = 20
    m = 10
    vec_val = np.zeros([m,1])
    mat_grad = np.zeros([n,m])
    for i in range(1,11):
        vec_val[i-1] = 0.5-x[i-1]
        mat_grad[i-1,i-1] = -1
    return vec_val,mat_grad







    
    
    
    
    
    
    
    