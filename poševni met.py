# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:58:38 2021

@author: mihap
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing()
from scipy.integrate import solve_ivp


H=1.5
m=0.5
v_0=30
g=9.81
alpha_0=30*np.pi/180

def fun_1(t, y):
    return y[1], 0, y[3], - g 

t = np.linspace(0, 4, 1000)
y_0 = np.array([0, v_0*np.cos(alpha_0), H, v_0*np.sin(alpha_0)])


rešitev = solve_ivp(fun_1, (0, 4), y_0, t_eval=t)

pomik_x = rešitev.y[0]
hitrost_x = rešitev.y[1]
pomik_y = rešitev.y[2]
hitrost_y = rešitev.y[3]

plt.plot(pomik_x, pomik_y, label='Pot')
plt.grid()
plt.legend();
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.ylim(0, 15);

for i in range(len(pomik_y)):
    if abs(pomik_y[i])<0.01:
        d=i


domet=pomik_x[d] 
t_k=t[d]

v_kx=hitrost_x[d]
v_ky=hitrost_y[d]
v_k=np.sqrt(v_ky**2+v_kx**2)

alpha_k=np.arctan(abs(v_kx/v_ky))

print(f'Domet krogle ={domet: .2f}m')
print(f'Čas letenja krogle = {t_k: .2f}s')
print(f'Hitrost, s katero krogla prileti na tla = {v_k: .2f}m/s')
print(f'Kot, pod katerim krogla prileti na tla = {alpha_k:.2f}rad({alpha_k*180/np.pi: .2f}°)')
