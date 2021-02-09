# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:28:12 2021

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
r=0.15
epsilon=0.2
mi=0.1
f=0.08
theta=f/r

def fun_1(t, y):
    """Popisuje gibanje pri poševnem metu"""
    return y[1], 0, y[3], - g 

a=100
t = np.linspace(0, a, 100000)

print(f'Poševni met in odboji')
x0=0
y0=H
vx0=v_0*np.cos(alpha_0)
vy0=v_0*np.sin(alpha_0)
skupen_x=np.array([])
skupen_y=np.array([])
skupen_t=np.array([])
t_k1=np.array([])
t_končni=np.array([0])
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
    t_vrednosti_pomika=t[:d1]+t_končni
    domet1=pomik_x[d1]
    v_kx1=hitrost_x[d1]
    v_ky1=hitrost_y[d1]
    v_k1=np.sqrt(v_ky1**2+v_kx1**2)
    alpha_k1=np.arctan(abs(v_kx1/v_ky1))
    v_ky2= -1*epsilon*v_ky1
    skupen_x=np.append(skupen_x,x_vrednosti_pomika)
    skupen_y=np.append(skupen_y,y_vrednosti_pomika)
    skupen_t=np.append(skupen_t,t_vrednosti_pomika)
    t_k1=t[d1]
    t_končni=t_končni+t_k1
    print(f'Domet do {j+1}. odboja je {domet1: .2f} m')
    print(f'Čas leta do {j+1}. odboja je {t_k1: .2f} s')
    print(f'Skupni čas je {t_končni[0]: .2f} s')
    if v_ky2>0.1:
        x0=pomik_x[d1]
        vx0=v_0*np.cos(alpha_0)
        y0= pomik_y[d1]
        vy0=v_ky2
        print(f'Hitrost vy po {j+1}. odboju je {v_ky2: .2f} m/s')
    else:
        domet_po_odbojih=domet1
        print(f'Hitrost vy bi bila po odboju {v_ky2: .2f} m/s')
        print(f'--------------------------------')
        print(f'Razdalja x po vseh odbojih je {domet_po_odbojih: .2f} m')
        print(f'--------------------------------')
        break

skupen_vx1=np.ones([len(skupen_y)])*vx0

x3 = np.linspace(0, skupen_x[-1], 3)
y3 = r+x3-x3
plt.plot(skupen_x, skupen_y, label='Položaj težišča krogle')
plt.plot(x3, y3, 'b--',label='Radij krogle')
plt.grid()
plt.legend();
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Gibanje krogle pri poševnem metu in odbojih')

def fun_2(t,y):
    """Za gibanje, ko se krogla vrti in drsi naenkrat"""
    return y[1], -mi*g, y[3], (5*mi*g)/(2*r)

y_02=np.array([domet_po_odbojih, vx0, 0, 0])
rešitev2 = solve_ivp(fun_2, (0, a), y_02, t_eval=t)

pomik_x2 = rešitev2.y[0]
hitrost_x2 = rešitev2.y[1]
pomik_phi2 = rešitev2.y[2]
hitrost_phi2 = rešitev2.y[3]

for i in range(len(hitrost_x2)):
        if abs(hitrost_x2[i]-hitrost_phi2[i]*r)<0.01:
            d2=i
            break
        
skupen_t2=t_končni[0]+t[:d2]
t_drsno=t[d2]
skupen_t=np.unique(np.append(skupen_t,skupen_t2))
t_končni2=t_končni[0]+t[d2]
x_končni2=pomik_x2[d2]

x_vrednosti_pomika2=pomik_x2[:d2]
skupen_x2=np.append(skupen_x,x_vrednosti_pomika2)
hitrost_po_drsnem=hitrost_x2[d2]
skupen_vx2=np.append(skupen_vx1,hitrost_x2[:d2])

print(f'Čas drsnega in kotalnega trenja je {t_drsno: .2f} s')
print(f'Skupen čas gibanja kroglice po drsnem trenju je {t_končni2: .2f} s')
print(f'Razdalja x po koncu drsnega trenja je {x_končni2: .2f} m')
print(f'Hitrost vx je po drsnem trenju {hitrost_po_drsnem: .2f} m/s')
print(f'--------------------------------')

def fun_3(t,y):
    """Za gibanje, ko se krogla kotali"""
    return y[1], -5/7*g*theta

y_03=np.array([x_končni2, hitrost_po_drsnem])
rešitev3 = solve_ivp(fun_3, (0, a), y_03, t_eval=t)
pomik_x3 = rešitev3.y[0]
hitrost_x3 = rešitev3.y[1]

for i in range(len(hitrost_x3)):
        if abs(hitrost_x3[i])<0.01:
            d3=i
            break

skupen_t3=t_končni2+t[:d3]
t_kotalno=t[d3]
skupen_t=np.unique(np.append(skupen_t,skupen_t3))
t_končni3=t_končni2+t[d3]
x_končni3=pomik_x3[d3]


x_vrednosti_pomika3=pomik_x3[:d3]
skupen_x3=np.append(skupen_x2,x_vrednosti_pomika3)
hitrost_po_kotalnem=hitrost_x3[d3]
skupen_vx3=np.append(skupen_vx2,hitrost_x3[:d3])

print(f'Čas kotalnega trenja je {t_kotalno: .2f} s')
print(f'Skupen čas gibanja kroglice je {t_končni3: .2f} s')
print(f'Razdalja x po koncu gibanja je {x_končni3: .2f} m')
print(f'Hitrost vx je po drsnem trenju {hitrost_po_kotalnem: .2f} m/s')


plt.plot(skupen_t,skupen_vx3, label='Hitrost vx težišča krogle')
plt.grid()
plt.legend();
plt.xlabel('t[s]');
plt.ylabel('vx[m/s]');
plt.title('Hitrost krogle');


plt.plot(skupen_t,skupen_x3, label='Položaj x-koordinate težišča krogle')
plt.grid()
plt.legend();
plt.xlabel('t[s]');
plt.ylabel('x[m]');
plt.title('Gibanje krogle na celotni poti');

plt.subplot(3,1,1)
plt.plot(skupen_x, skupen_y, label='Položaj težišča krogle')
plt.plot(x3, y3, 'b--',label='Radij krogle')
plt.grid()
plt.legend();
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Gibanje krogle pri poševnem metu in odbojih');
plt.subplot(3 ,1, 2)
plt.plot(skupen_t,skupen_vx3, label='Hitrost vx težišča krogle')
plt.grid()
plt.legend();
plt.xlabel('t[s]');
plt.ylabel('vx[m/s]');

plt.subplot(3, 1, 3)
plt.plot(skupen_t,skupen_x3, label='Položaj x-koordinate težišča krogle')
plt.grid()
plt.legend();
plt.xlabel('t[s]');
plt.ylabel('x[m]');
