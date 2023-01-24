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
    return math.sqrt(sum(vec**2))

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

    return upper,-center,-lower,val0,norm_center2,norm_lower2

def update(n,x,lambd,nu,A,b,m,l,eval_f0,eval_f_ineq,H_lambda_nu,H_x,rho,s,T_lambda_nu,T_x):
    
    S_lambda_nu = -np.matmul(H_lambda_nu,T_lambda_nu)
    S_x = -np.matmul(H_x,T_x)
    x += S_x
    lambd += S_lambda_nu[0:m]
    nu += S_lambda_nu[m:]
    newT_x,newT_lambda,newT_nu,Val,norm_T_lambda,norm_T_nu= get_T(n,x,lambd,nu,A,b,m,l,
                                                          eval_f0,eval_f_ineq,rho,s)
    newT_lambda_nu  = np.concatenate((newT_lambda,newT_nu),axis = 0)
    
    y_x = newT_x-T_x
    y_lambda_nu = newT_lambda_nu-T_lambda_nu
    
    I_x = np.identity(n)
    I_lambda_nu = np.identity(m+l)
    
    w_x=np.matmul(y_x.transpose(),S_x)
    L_x=I_x-np.matmul(S_x,y_x.transpose())/w_x
    H_x = np.matmul(np.matmul(L_x,H_x),L_x.transpose()) + np.matmul(S_x,S_x.transpose())/w_x 
    
    w_lambda_nu=np.matmul(y_lambda_nu.transpose(),S_lambda_nu)
    L_lambda_nu=I_lambda_nu-np.matmul(S_lambda_nu,y_lambda_nu.transpose())/w_lambda_nu
    H_lambda_nu = np.matmul(np.matmul(L_lambda_nu,H_lambda_nu),L_lambda_nu.transpose()) + np.matmul(S_lambda_nu,S_lambda_nu.transpose())/w_lambda_nu                                    
    
    return x,lambd,nu,Val,norm_T_lambda,norm_T_nu,H_x,H_lambda_nu,newT_x,newT_lambda_nu


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

def run(A,b,x,nu,lambd,m,eval_f0,eval_f_ineq,Print=False,history=False,delta=0.5,rho=1,s=1.5,numiter=200,eps=0.001):
    n=A.shape[1]
    l = A.shape[0]
    FEAS = []
    VAL = []
    x_axis = []
    x_star=x
    val_star=eval_f0(x)[0]
    
    H_x=np.identity(n)
    H_lambda_nu = np.identity(m+l)
    
    if Print:
        print('PDS: version 1.0')
        print('Number of variables: n = {} '.format(n))
        print('Number of enequality constrains: m = {}'.format(m))
        print('Number of equality constrains: l = {}'.format(l))
    t = time.time()
    i=0
    
    T_x,T_lambda,T_nu,_,_,_= get_T(n,x,lambd,nu,A,b,m,l,eval_f0,eval_f_ineq,rho,s)
    T_lambda_nu = np.concatenate((T_lambda,T_nu),axis = 0)
    
    for k in range(1,numiter):

        
        x,lambd,nu,Val,norm_T_lambda,norm_T_nu,H_x,H_lambda_nu,newT_x,newT_lambda_nu = update(n,x,lambd,nu,A,
                                                                                            b,m,l,eval_f0,
                                                                                            eval_f_ineq,H_lambda_nu,
                                                                                            H_x,rho,s,T_lambda_nu,
                                                                                            T_x)
        feas = norm_T_lambda + norm_T_nu
        

        
        if history:
               
                FEAS.append(feas)
                VAL.append(Val)
    
        
        if k==2**i or k==numiter-1:
        
            if Print:
                print('Iter = {},  Val = {:.4f},  InFeas = {:.4f}'.format(k,Val,feas))
            i+=1
    if Print:
        print('Total time: {:.4f} '.format(time.time()-t))
        
    return Val,x,FEAS,VAL