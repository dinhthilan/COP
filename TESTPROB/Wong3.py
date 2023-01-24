# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:25:05 2020

@author: Thi Lan Dinh
"""
import numpy as np

def f1(x):
    val= x[0]**2+x[1]**2+x[0]*x[1]-14*x[0]-16*x[1]+(x[2]-10)**2+4*(x[3]-5)**2+(x[4]-3)**2+2*(x[5]-1)**2+5*x[6]**2
    +7*(x[7]-11)**2+2*(x[8]-10)**2+(x[9]-7)**2+(x[10]-9)**2+10*(x[11]-1)**2+5*(x[12]-7)**2+4*(x[13]-14)**2
    +27*(x[14]-1)**2+x[15]**4+(x[16]-2)**2+13*(x[17]-2)**2+(x[18]-3)**2+x[19]**2+95
    
    grad=np.array([2*x[0]+x[1]-14,
                   2*x[1] + x[0]-16,
                   2*(x[2]-10),
                   8*(x[3]-5),
                   2*(x[4]-3),
                   4*(x[5]-1),
                   10*x[6],
                   14*(x[7]-11),
                   4*(x[8]-10),
                   2*(x[9]-7),
                   2*(x[10]-9),
                   20*(x[11]-1),
                   10*(x[12]-7),
                   8*(x[13]-14),
                   54*(x[14]-1),
                   4*x[15]**3,
                   2*(x[16]-2),
                   26*(x[17]-2),
                   2*(x[18]-3),
                   2*x[19]])
    return val,grad.reshape(20,)

         

def term_f2(x):
    val= 10*(3*(x[0]-2)**2+4*(x[1]-3)**2+2*x[2]**2-7*x[3]-120)
    grad = np.zeros([20,1])
    
    grad[0]=6*(x[0]-2)
    grad[1]=8*(x[1]-3)
    grad[2]=4*x[2]
    grad[3]=-7
    grad*=10

    return val,grad.reshape(20,)

def term_f3(x):
    val=10*(5*x[0]**2+8*x[1]+(x[2]-6)**2-2*x[3]-40)
    grad = np.zeros([20,1])
    grad[0]=10*x[0]
    grad[1]= 8
    grad[2]=  2*(x[2]-6)
    grad[3]=-2
    grad*=10
    return val,grad.reshape(20,)

def term_f4(x):
    val=10*(0.5*(x[0]-8)**2+2*(x[1]-4)**2+3*x[4]**2-x[5]-30)
    grad = np.zeros([20,1])
    grad[0]=x[0]-8
    grad[1]= 4*(x[1]-4)
    grad[4]= 6*x[4]
    grad[5]= -1
      
    grad*=10
    return val,grad.reshape(20,)

def term_f5(x):
    val=10*(x[0]**2+2*(x[1]-2)**2-2*x[0]*x[1]+14*x[4]-6*x[5])
    grad = np.zeros([20,1])
    grad[0]=2*x[0]-2*x[1]
    grad[1]=4*(x[1]-2)-2*x[0]
    grad[4]=14
    grad[5]=-6
    grad*=10
    return val,grad.reshape(20,)

def term_f6(x):
    val=10*(3*x[0]+6*x[1]+12*(x[8]-8)**2-7*x[9])
    grad = np.zeros([20,1])
    grad[0]=3
    grad[1]=6
    grad[8]=24*(x[8]-8)
    grad[9]=-7
    grad*=10
    return val,grad.reshape(20,)

def term_f7(x):
    val=10*(x[0]**2+15*x[10]-8*x[11]-28)
    grad= np.zeros([20,1])
    grad[0] = 2*x[0]
    grad[10] = 15
    grad[11] = -8
    grad*=10
    return val,grad.reshape(20,)

def term_f8(x):
    val=10*(4*x[0]+9*x[1]+5*x[12]**2-9*x[13]-87)
    grad= np.zeros([20,1])
    grad[0] = 4
    grad[1] = 9
    grad[12] = 10*x[12]
    grad[13] = -9
    grad*=10
    return val,grad.reshape(20,)

def term_f9(x):
    val=10*(3*x[0]+4*x[1]+3*(x[12]-6)**2-14*x[13]-10)
    grad= np.zeros([20,1])
    grad[0] = 3
    grad[1] = 4
    grad[12] = 6*(x[12]-6)
    grad[13] = -14
    grad*=10
    return val,grad.reshape(20,)

def term_f10(x):
    val=10*(14*x[0]**2+35*x[14]-79*x[15]-92)
    grad= np.zeros([20,1])
    grad[0] = 28*x[0]
    grad[14] = 35
    grad[15] = -79
    grad*=10
    return val,grad.reshape(20,)

def term_f11(x):
    val=10*(15*x[1]**2+11*x[14]-61*x[15]-54)
    grad= np.zeros([20,1])
    grad[1] = 30*x[1]
    grad[14] = 11
    grad[15] = -61
    grad*=10
    return val,grad.reshape(20,)

def term_f12(x):
    val=10*(5*x[0]**2+2*x[1]+9*x[16]**4-x[17]-68)
    grad= np.zeros([20,1])
    grad[0] = 10*x[0]
    grad[1] = 2
    grad[16] = 36*x[16]**3
    grad[17] = -1
    grad*=10
    return val,grad.reshape(20,)

def term_f13(x):
    val=10*(x[0]**2-x[1]+19*x[18]-20*x[19]+19)
    grad= np.zeros([20,1])
    grad[0] = 2*x[0]
    grad[1] = -1
    grad[18] = 19
    grad[19] = -20
    grad*=10
    return val,grad.reshape(20,)

def term_f14(x):
    val=10*(7*x[0]**2+5*x[1]**2+x[18]**2-30*x[19])
    grad= np.zeros([20,1])
    grad[0] = 14*x[0]
    grad[1] = 10*x[1]
    grad[18] = 2*x[18]
    grad[19] = -30
    grad*=10
    return val,grad.reshape(20,)

def c1(x):
    val=4*x[0]+5*x[1]-3*x[6]+9*x[7]-105
    grad = np.zeros([20,])
    grad[0] = 4
    grad[1] = 5
    grad[6] = -3
    grad[7] = 9
    return val, grad

def c2(x):
    val=10*x[0]-8*x[1]-17*x[6]+2*x[7]
    grad = np.zeros([20,])
    grad[0] = 10
    grad[1] = -8
    grad[6] = -17
    grad[7] = 2
    return val,grad
    
def c3(x):
    val=-8*x[0]+2*x[1]+5*x[8]-2*x[9]-12
    grad = np.zeros([20,])
    grad[0] = -8
    grad[1] = 2
    grad[8] = 5
    grad[9] = -2
    return val,grad

def c4(x):
    val=x[0]+x[1]+4*x[10]-21*x[11]
    grad = np.zeros([20,])
    grad[0] = 1
    grad[1] = 1
    grad[10] = 4
    grad[11] = -21
    return val,grad


def max_func(n,vec_val,mat_grad):
    indx = np.argmax(vec_val)
    val = vec_val[indx]
    grad = mat_grad[:,indx]
    return val[0],grad.reshape(n,1)
def obj(x):
    n = 20
    m = 14
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
    
    vec_val[6],mat_grad[:,6] = term_f7(x)
    vec_val[6]+= vec_val[0]
    mat_grad[:,6]+= mat_grad[:,0]
    
    vec_val[7],mat_grad[:,7] = term_f8(x)
    vec_val[7]+= vec_val[0]
    mat_grad[:,7]+= mat_grad[:,0]
    
    vec_val[8],mat_grad[:,8] = term_f9(x)
    vec_val[8]+= vec_val[0]
    mat_grad[:,8]+= mat_grad[:,0]
    
    vec_val[9],mat_grad[:,9] = term_f10(x)
    vec_val[9]+= vec_val[0]
    mat_grad[:,9]+= mat_grad[:,0]
    
    vec_val[10],mat_grad[:,10] = term_f11(x)
    vec_val[10]+= vec_val[0]
    mat_grad[:,10]+= mat_grad[:,0]
    
    vec_val[11],mat_grad[:,11] = term_f12(x)
    vec_val[11]+= vec_val[0]
    mat_grad[:,11]+= mat_grad[:,0]
    
    vec_val[12],mat_grad[:,12] = term_f13(x)
    vec_val[12]+= vec_val[0]
    mat_grad[:,12]+= mat_grad[:,0]
    
    vec_val[13],mat_grad[:,13] = term_f14(x)
    vec_val[13]+= vec_val[0]
    mat_grad[:,13]+= mat_grad[:,0]
    
    return max_func(n,vec_val,mat_grad)
    

def cons(x):
    n = 20
    m = 4
    vec_val = np.zeros([m,1])
    mat_grad = np.zeros([n,m])
    vec_val[0],mat_grad[:,0] = c1(x)
    vec_val[1],mat_grad[:,1] = c2(x)
    vec_val[2],mat_grad[:,2] = c3(x)
    vec_val[3],mat_grad[:,3] = c4(x)
    return vec_val,mat_grad
