# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:10:55 2020

@author: Thi Lan Dinh
"""
import numpy as np
import math

def rand_mat(m,n):
    return  2*np.random.rand(m,n)-1
    
def linear_obj(c,x,n):
    return np.dot(c,x.reshape(n,1))[0],c.reshape(n,1)

def max_func(n,vec_val,mat_grad):
    indx = np.argmax(vec_val)
    val = vec_val[indx]
    grad = mat_grad[:,indx]
    return val[0],grad.reshape(n,)

def l1_norm(variant):
    subgrad = np.sign(variant)
    val =  np.linalg.norm(variant,ord = 1)   
    return val.reshape(1,1)-1,subgrad

def cons1(x):
    return l1_norm(x)

def minus_log(n,x):
    val = -np.log(x[0]+1)
    subgrad = np.zeros([n,1])
    subgrad[0] = -1/(x[0]+1)
    return val,subgrad.reshape(n,)

def max_minus_log_x2(n,x):
    vec_val = np.zeros([2,1])
    mat_grad = np.zeros([n,2])
    vec_val[0],mat_grad[:,0] = minus_log(n,x)
    vec_val[1] = x[1]
    mat_grad[1,1]=1    
    return max_func(n,vec_val-1,mat_grad)

def cons2(n,x):
    vec_val = np.zeros([2*n+1,1])
    mat_grad = np.zeros([n,2*n+1])
    
    for i in range(0,n):
        vec_val[i] = x[i]-1
        vec_val[i+n]= -x[i]-1
        mat_grad[i,i] = 1
        mat_grad[i,i+n] = -1
    vec_val[2*n-1],mat_grad[:,2*n-1] = max_minus_log_x2(n,x)
    return vec_val,mat_grad


def l_inf(n,x):
    vec_val = np.zeros([n,1])
    mat_grad = np.zeros([n,n])
    for j in range(n):
        vec_val[j]=abs(x[j])-1
        mat_grad[j,j] = np.sign(x[j])
    return max_func(n,vec_val,mat_grad)
    

def exp(n,x):
    val = math.exp(x[0]+x[1])-x[2]
    subgrad = np.zeros([n,1])
    subgrad[0] = math.exp(x[0]+x[1])
    subgrad[1] = math.exp(x[0]+x[1])
    subgrad[2] = -1
    return val,subgrad.reshape(n,)


def cons3(n,x):
    vec_val = np.zeros([2,1])
    mat_grad = np.zeros([n,2])
    vec_val[0],mat_grad[:,0] = l_inf(n,x)
    vec_val[1],mat_grad[:,1] = exp(n,x)
    return vec_val,mat_grad


def cons4(n,x):
    vec_val = np.zeros([2*n,1])
    mat_grad = np.zeros([n,2*n])
    
    for i in range(0,n):
        vec_val[i] = x[i]-1
        vec_val[i+n]= -x[i]-1
        mat_grad[i,i] = 1
        mat_grad[i,i+n] = -1
   
    return vec_val,mat_grad
















    