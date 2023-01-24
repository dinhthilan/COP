# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:48:06 2020

@author: Lan-ACSYON
"""
import matplotlib.pyplot as plt

def plot(list_iter,Obj,Labels,check = 'feas'):
    
    fig,ax=plt.subplots()
    for i in range(len(Obj)):
        ax.plot(list_iter,Obj[i],'--',linewidth=1,label=Labels[i])
        ax.set_xscale('log',base=2)
        plt.grid(True)

    plt.xlabel('Iterations')
    if check == 'feas':
        plt.ylabel('Infeasibility')
    else:
        plt.ylabel('Value')
    legend = ax.legend(loc='upper right')   
    plt.show()