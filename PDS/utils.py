# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:07:59 2020

@author: Thi Lan Dinh
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import time




def Axb(x,A,b):
    return np.matmul(A,x)-b

def normsq(vec):
    return sum(vec**2)

def checkzero(vec,norm_vec2,l,s):
    if norm_vec2 == 0:
        return np.zeros([l,1])
    else: return s*norm_vec2**(0.5*(s-2))*vec
    
    
def get_T(n,x,lambd,v,A,b,m,l,eval_f0,eval_f_ineq,rho,s):
    val0,g0 = eval_f0(x)
    
    center,gF = eval_F_new(m,n,eval_f_ineq,x)
    lower = Axb(x,A,b)
    norm_center2=normsq(center)
    norm_lower2=normsq(lower)     
    upper = g0 + np.matmul(gF,lambd + rho*checkzero(center,norm_center2,m,s))+np.matmul(A.transpose(),v+rho*checkzero(lower,norm_lower2,l,s))

    return upper,center,lower,val0,norm_center2[0],norm_lower2


    
def update(x,x_star,val_star,lambd,v,A,b,m,n,l,k,eval_f0,eval_f_ineq,delta=1,rho=1,s=1.5,eps=0.001):
    up,cen,low,Val,norm_center2,norm_lower2= get_T(n,x,lambd,v,A,b,m,l,eval_f0,eval_f_ineq,rho,s)
    normT2=math.sqrt(norm_center2+norm_lower2+sum(up**2))
    feas = math.sqrt(norm_center2)+math.sqrt(norm_lower2)
    
    alpha = k**(-1+delta/2)/normT2
    if feas< eps and Val < val_star:
        x_star = x
        val_star= Val
    return x-alpha*up,x_star,lambd + alpha*cen,v + alpha*low,Val,val_star,feas,normT2


def eval_max_with_zero(m,n,val,subgrad):
    val_new,subgrad_new=0,np.zeros([n,1])
    if val>0:
        val_new,subgrad_new=val,subgrad
    return val_new,subgrad_new.reshape(n,)

def eval_maxvec(m,n,vec_val,mat_grad):
    for j in range(0,m):
        vec_val[j],mat_grad[:,j]=eval_max_with_zero(m,n,vec_val[j],mat_grad[:,j])
    return vec_val.reshape(m,1),mat_grad

def eval_F_new(m,n,eval_f_ineq,x):
    vec_val,mat_grad=eval_f_ineq(x)
    return eval_maxvec(m,n,vec_val,mat_grad)


def gap(val1,val2):
    return abs(val1-val2)/(1+max(abs(val1),abs(val2)))

def run(A,b,x,v,lambd,m,eval_f0,eval_f_ineq,Print=False,history=False,delta=0.5,rho=1,s=1.5,numiter=200,eps=0.001):
    n=A.shape[1]
    l = A.shape[0]
    FEAS = []
    VAL = []
    x_axis = []
    x_star=x
    val_star=eval_f0(x)[0]
    
    if Print:
        print('PDS: version 1.0')
        print('Number of variables: n = {} '.format(n))
        print('Number of enequality constrains: m = {}'.format(m))
        print('Number of equality constrains: l = {}'.format(l))
    t = time.time()
    i=0
    for k in range(1,numiter):

        
        x,x_star,lambd,v,Val,Val_star,feas,normT2 = update(x,x_star,val_star,lambd,v,A,b,m,n,l,k
                                                          ,eval_f0,eval_f_ineq,
                                                          delta=delta,rho=rho,s=s,eps=eps)
        
        

        
        if history:
               
                FEAS.append(feas)
                VAL.append(Val)
    
        
        if k==2**i or k==numiter-1:
        
            if Print:
                print('Iter = {},  Val = {:.4f},  InFeas = {:.4f}'.format(k,Val,feas))
            i+=1
    if Print:
        print('Total time: {:.4f} '.format(time.time()-t))
        
    return Val_star,x_star,Val,x,FEAS,VAL