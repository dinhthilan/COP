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

def max_with_zero(m,n,val,subgrad):
    val_new,subgrad_new=0,np.zeros([n,1])
    if val>0:
        val_new,subgrad_new=val,subgrad
    return val_new,subgrad_new.reshape(n,)

def maxvec(m,n,vec_val,mat_grad):
    for j in range(0,m):
        vec_val[j],mat_grad[:,j]=max_with_zero(m,n,vec_val[j],mat_grad[:,j])
    return vec_val.reshape(m,1),mat_grad

def max_f_ineq_with_zero(n,m,f_ineq,x):
    vec_val,mat_grad=f_ineq(x)
    return maxvec(m,n,vec_val,mat_grad)



def update(n,m,l,x_hat,s_hat,x0,mu0,theta0,x,mu,theta,beta,s_x,s_mu,s_theta,f_ineq,h_eq,obj,rho):
    val_obj,grad_obj = obj(x)
    vec_val_ineq,mat_grad_ineq = max_f_ineq_with_zero(n,m,f_ineq,x)
    vec_val_eq,mat_grad_eq = h_eq(x)
    G_x = grad_obj+ np.matmul(mat_grad_ineq,mu+2*rho*vec_val_ineq.reshape(m,))+np.matmul(mat_grad_eq,theta+2*rho*vec_val_eq.reshape(l,))
    #G_x = grad_obj+ np.matmul(mat_grad_ineq,mu)+np.matmul(mat_grad_eq,theta)
    G_mu = vec_val_ineq.reshape(m,)
    G_theta = vec_val_eq.reshape(l,)
   
    norm_G = math.sqrt(sum(G_x**2)+sum(G_mu**2)+sum(G_theta)**2)
    s_x += G_x/norm_G
    s_mu -= G_mu/norm_G
    s_theta -= G_theta/norm_G
    
    
    s_hat += 1/norm_G
    x_hat += x/norm_G
    
    x = x0-s_x/beta
    mu = mu0 -s_mu/beta
    theta = theta0 -s_theta/beta
    beta += 1/beta
    return x_hat,s_hat,x,mu,theta,beta,s_x,s_mu,s_theta

def plot(epoch,obj,check = 'Infeas'):
    
    fig,ax=plt.subplots()
    ax.plot(range(1,epoch),obj,label='MDSG')
    ax.set_xscale('log',base=2)
    plt.xlabel('Iterations')
    if check == 'Infeas':
        plt.ylabel('Feasibility')
    else:
        plt.ylabel('Value')
    
    legend = ax.legend(loc='upper right')   
    plt.show()

def run(n,m,l,x_hat,s_hat,x0,mu0,theta0,beta,s_x,s_mu,s_theta,f_ineq,h_eq,obj,rho=0.5,Print=False,history=False,numiter=200):
   
    x,mu,theta=x0,mu0,theta0
    FEAS = []
    VAL = []

    
    if Print:
        print('MultiDSG: version 1.0')
        print('Number of variables: n = {} '.format(n))
        print('Number of enequality constrains: m = {}'.format(m))
        print('Number of equality constrains: l = {}'.format(l))
    
    t = time.time()
    i=0
    for k in range(1,numiter):

        
        x_hat,s_hat,x,mu,theta,beta,s_x,s_mu,s_theta = update(n,m,l,x_hat,s_hat,x0,mu0,theta0,x,mu,theta,beta,s_x,s_mu,s_theta,f_ineq,h_eq,obj,rho)
        
        x_bar = x_hat/s_hat
        

        Val = obj(x_bar)[0]
        Infeas = math.sqrt(sum(max_f_ineq_with_zero(n,m,f_ineq,x)[0]**2)+sum(h_eq(x)[0]**2))
                           
        
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


   