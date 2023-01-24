# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:53:23 2020

@author: Thi Lan Dinh
"""
import numpy as np
def f1(x):
    val=x[0]**2+x[1]**2+x[0]*x[1]-14*x[0]-16*x[1]+(x[2]-10)**2+4*(x[3]-5)**2+(x[4]-3)**2+2*(x[5]-1)**2+5*x[6]**2+7*(x[7]-11)**2+2*(x[8]-10)**2+(x[9]-7)**2+45
    grad=np.array([2*x[0]+x[1]-14,
                   2*x[1] + x[0]-16,
                   2*(x[2]-10),
                   8*(x[3]-5),
                   2*(x[4]-3),
                   4*(x[5]-1),
                   10*x[6],
                   14*(x[7]-11),
                   4*(x[8]-10),
                   2*(x[9]-7)])
    return val,grad.reshape(10,)

         

def term_f2(x):
    val= 10*(3*(x[0]-2)**2+4*(x[1]-3)**2+2*x[2]**2-7*x[3]-120)
    grad = np.array([6*(x[0]-2),
                          8*(x[1]-3),
                          4*x[2],
                          -7,
                          0,
                          0,
                          0,
                          0,
                          0,
                          0])
    grad*=10

    return val,grad.reshape(10,)

def term_f3(x):
    val=10*(5*x[0]**2+8*x[1]+(x[2]-6)**2-2*x[3]-40)
    grad=np.array([10*x[0],
                          8,
                          2*(x[2]-6),
                          -2,
                          0,
                          0,
                          0,
                          0,
                          0,
                          0])
    grad*=10
    return val,grad.reshape(10,)

def term_f4(x):
    val=10*(0.5*(x[0]-8)**2+2*(x[1]-4)**2+3*x[4]**2-x[5]-30)
    grad=np.array([x[0]-8,
                          4*(x[1]-4),
                          0,
                          0,
                          6*x[4],
                          -1,
                          0,
                          0,
                          0,
                          0])
    grad*=10
    return val,grad.reshape(10,)

def term_f5(x):
    val=10*(x[0]**2+2*(x[1]-2)**2-2*x[0]*x[1]+14*x[4]-6*x[5])
    grad=np.array([2*x[0]-2*x[1],
                          4*(x[1]-2)-2*x[0],
                          0,
                          0,
                          14,
                          -6,
                          0,
                          0,
                          0,
                          0])
    grad*=10
    return val,grad.reshape(10,)

def term_f6(x):
    val=10*(-3*x[0]+6*x[1]+12*(x[8]-8)**2-7*x[9])
    grad=np.array([-3,
                          6,
                          0,
                          0,
                          0,
                          0,
                          0,
                          0,
                          24*(x[8]-8),
                          -7])
    grad*=10
    return val,grad.reshape(10,)

def c1(x):
    val=4*x[0]+5*x[1]-3*x[6]+9*x[7]-105
    grad=np.array([4,5,0,0,0,0,-3,9,0,0]).reshape(10,)
    return val, grad

def c2(x):
    val=10*x[0]-8*x[1]-17*x[6]+2*x[7]
    grad=np.array([10,-8,0,0,0,0,-17,2,0,0]).reshape(10,)
    return val,grad
    
def c3(x):
    val=-8*x[0]+2*x[1]+5*x[8]-2*x[9]-12
    grad=np.array([-8,2,0,0,0,0,0,0,5,-2]).reshape(10,)
    return val,grad

def max_func(n,vec_val,mat_grad):
    indx = np.argmax(vec_val)
    val = vec_val[indx]
    grad = mat_grad[:,indx]
    return val[0],grad.reshape(n,1)
def obj(x):
    n = 10
    m = 6
    vec_val = np.zeros([m,1])
    mat_grad = np.zeros([n,m])
    
           
   
    vec_val[0],mat_grad[:,0] = f1(x)
    vec_val[1],mat_grad[:,1] = term_f2(x)
    vec_val[1]+= vec_val[0]
    mat_grad[:,1]+= mat_grad[:,0]
    

    vec_val[2],mat_grad[:,2] = term_f3(x)
    vec_val[2]+= vec_val[0]
    mat_grad[:,2]+= mat_grad[:,0]
    
    vec_val[3],mat_grad[:,3] = term_f4(x)
    vec_val[3]+= vec_val[0]
    mat_grad[:,3]+= mat_grad[:,0]
    
    vec_val[4],mat_grad[:,4] = term_f5(x)
    vec_val[4]+= vec_val[0]
    mat_grad[:,4]+= mat_grad[:,0]

    vec_val[5],mat_grad[:,5] = term_f6(x)
    vec_val[5]+= vec_val[0]
    mat_grad[:,5]+= mat_grad[:,0]
    
    return max_func(n,vec_val,mat_grad)
    

def cons(x):
    n = 10
    m = 3
    vec_val = np.zeros([m,1])
    mat_grad = np.zeros([n,m])
    vec_val[0],mat_grad[:,0] = c1(x)
    vec_val[1],mat_grad[:,1] = c2(x)
    vec_val[2],mat_grad[:,2] = c3(x)
    return vec_val,mat_grad
