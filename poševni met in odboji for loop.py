# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:28:12 2021

@author: mihap
"""

import numpy as np
from ipywidgets import interact
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing()
from scipy.integrate import solve_ivp

H=1.5
m=0.5
v_0=30
g=9.81
alpha_0=30*np.pi/180
r=0.15
epsilon=0.2

def fun_1(t, y):
    return y[1], 0, y[3], - g 

a=100
t = np.linspace(0, a, 100000)

x0=0
y0=H
vx0=v_0*np.cos(alpha_0)
vy0=v_0*np.sin(alpha_0)
skupen_x=np.array([])
skupen_y=np.array([])
for j in range(50):
    y_0=np.array([x0,vx0,y0,vy0])
    rešitev = solve_ivp(fun_1, (0, a), y_0, t_eval=t)
    pomik_x = rešitev.y[0]
    hitrost_x = rešitev.y[1]
    pomik_y = rešitev.y[2]
    hitrost_y = rešitev.y[3]
    for i in range(len(pomik_y)):
        if abs(pomik_y[i])<0.001+r and hitrost_y[i]<0:
            d1=i
            break
    x_vrednosti_pomika=pomik_x[:d1]
    y_vrednosti_pomika=pomik_y[:d1]
    domet1=pomik_x[d1]
    v_kx1=hitrost_x[d1]
    v_ky1=hitrost_y[d1]
    v_k1=np.sqrt(v_ky1**2+v_kx1**2)
    alpha_k1=np.arctan(abs(v_kx1/v_ky1))
    t_k1=t[d1]
    v_ky2= -1*epsilon*v_ky1
    skupen_x=np.append(skupen_x,x_vrednosti_pomika)
    skupen_y=np.append(skupen_y,y_vrednosti_pomika)
    print(f'Domet do {j+1}. odboja je {domet1: .2f} m')
    print(f'Čas leta do {j+1}. odboja je {t_k1: .2f} s')
    if v_ky2>0.1:
        x0=pomik_x[d1]
        vx0=v_0*np.cos(alpha_0)
        y0= pomik_y[d1]
        vy0=v_ky2
        print(f'Hitrost vy po {j+1}. odboju je {v_ky2: .2f} m/s')
    else:
        print(f'Hitrost vy bi bila po odboju {v_ky2: .2f} m/s')
        break

x3 = np.linspace(0, skupen_x[-1], 3)
y3 = r+x3-x3
plt.plot(skupen_x, skupen_y, label='Položaj težišča krogle')
plt.plot(x3, y3, 'b--',label='Radij krogle')
plt.grid()
plt.legend();
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Gibanje krogle')