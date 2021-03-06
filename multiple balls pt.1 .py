# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:02:32 2021

@author: mihap
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import time 
sym.init_printing()
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


def fun_projectile_motion(t, y):
    """
    Function is used to solve differential equations,
    while the object is in the air.
    """
    return y[1], 0, y[3], - g


def fun_slipping_rolling(t, y):
    """
    Function is used to solve differential equations,
    while the object is slipping and rolling.
    """
    return y[1], -mi * g, y[3], (5 * mi * g) / (2 * radius)


def fun_rolling(t, y):
    """
    Function is used to solve differential equations,
    while the object is rolling.
    """
    return y[1], -5 / 7 * g * theta


class World:
    def __init__(self, max_balls):
        self.max_balls = max_balls
        self.balls = []
        
    def add_ball(self, ball):
        if len(self.balls) < self.max_balls:
            self.balls.append(ball)
            return True
        return False
    
    def get_balls(self):
        return self.balls

class Ball:
    
    def __init__(self, ID, distance_x, velocity_x,distance_y,velocity_y, distance_phi, velocity_phi,radius,mass):
        self.ID = ID
        self.distance_x = distance_x
        self.velocity_x = velocity_x
        self.distance_y = distance_y
        self.velocity_y = velocity_y
        self.distance_phi = distance_phi
        self.velocity_phi = velocity_phi
        self.radius = radius
        self.mass = mass
        #self.motion_status = motion_status

        
    def get_ID(self):
        return self.ID
        
    def get_distance_x(self):
        return self.distance_x
    
    def get_velocity_x(self):
        return self.velocity_x
    
    def get_distance_y(self):
        return self.distance_y
    
    def get_velocity_y(self):
        return self.velocity_y
    
    def get_distance_phi(self):
        return self.distance_phi
    
    def get_velocity_phi(self):
        return self.velocity_phi
    
    def get_radius(self):
        return self.radius
    
    def get_mass(self):
        return self.mass
    
    
    def set_distance_x(self,distance_x):
        self.distance_x = distance_x
    
    def set_velocity_x(self, velocity_x):
        self.velocity_x = velocity_x
        
    def set_distance_y(self,distance_y):
        self.distance_y = distance_y
        
    def set_velocity_y(self, velocity_y):
        self.velocity_y = velocity_y
        
    def set_distance_phi(self,distance_phi):
        self.distance_phi = distance_phi
        
    def set_velocity_phi(self, velocity_phi):
        self.velocity_phi = velocity_phi
        
    def set_radius(self, radius):
        self.radius = radius
    
    def set_mass(self, mass):
        self.mass = mass
        
    
    def get_motion_status(self):
        """
        Function used to determine motion state of the ball.
        """
        if self.velocity_x == 0 and self.distance_y == self.radius:
            state = "not_moving"
        elif self.distance_y > self.radius or self.velocity_y > 0:
            state = "projectile_motion"
        elif self.distance_y < self.radius and abs(self.velocity_y) > 0:
            state = "collision"
        elif self.velocity_y == 0 and self.distance_y== self.radius and abs(self.velocity_x) - abs(self.velocity_phi) * self.radius > 0.001:
            state = "sliding"
        elif self.velocity_y == 0 and self.distance_y == self.radius and abs(self.velocity_x) - abs(self.velocity_phi) * self.radius < 0.001 and abs(self.velocity_x) > 0:
            state = "rolling"
        else:
            print('Error at determinating motion state!')
            
    
        return state
    """Used for  calculating distances after collision"""
    def set_time_after_collision(self, time_after_collision):
        self.time_after_collision = time_after_collision
    def get_time_after_collision(self):
        return self.time_after_collision
    def set_time_after_sliding(self, time_after_sliding):
        self.time_after_sliding = time_after_sliding
    def get_time_after_sliding(self):
        return self.time_after_sliding
    def set_time_after_rolling(self, time_after_rolling):
        self.time_after_rolling = time_after_rolling
    def get_time_after_rolling(self):
        return self.time_after_rolling
    def set_distance_phi_after_sliding(self,distance_phi_after_sliding):
        self.distance_phi_after_sliding = distance_phi_after_sliding
    def get_distance_phi_after_sliding(self):
        return self.distance_phi_after_sliding
    def set_distance_x_after_sliding(self,distance_x_after_sliding):
        self.distance_x_after_sliding = distance_x_after_sliding
    def get_distance_x_after_sliding(self):
        return self.distance_x_after_sliding
    def set_distance_phi_after_rolling(self,distance_phi_after_rolling):
        self.distance_phi_after_rolling = distance_phi_after_rolling
    def get_distance_phi_after_rolling(self):
        return self.distance_phi_after_rolling
    def set_velocity_x_after_sliding(self,velocity_x_after_sliding):
        self.velocity_x_after_sliding = velocity_x_after_sliding
    def get_velocity_x_after_sliding(self):
        return self.velocity_x_after_sliding
    
    
    
        
    
        



def time_solution(ball):
    values_array = []

    
    for i in range(len(time_array) - 1):
        for j in range(1):
            
            if ball.get_motion_status() == 'projectile_motion':
                
                y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(), ball.get_velocity_y()])
                solution = solve_ivp(fun_projectile_motion, (time_array[i], time_array[i + 1]), y_0)
                ball.set_distance_x(solution.y[0][-1])
                ball.set_velocity_x(solution.y[1][-1])
                ball.set_distance_y(solution.y[2][-1])
                ball.set_velocity_y(solution.y[3][-1])
                
                """if ball.get_distance_y() < ball.get_radius():
                    pass
                else:
                
                    current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                    values_array.append(current_values)""" # problem nastane, ker ni time_array enake velikosti kot distance_array
                current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                values_array.append(current_values)
                
    
            elif ball.get_motion_status() == 'collision':
                if abs(ball.get_velocity_y() * epsilon) >= 0.1:
                    ball.set_velocity_y(ball.get_velocity_y() * epsilon * (-1))
                    ball.set_distance_y(radius)
                    current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                    values_array.append(current_values)
        
                elif abs(ball.get_velocity_y() * epsilon) < 0.1:
                    print(f'Time elapsed after last collision is{i*a/b: .2f} s.')
                    print(f'Distance x after last collision is{ball.get_distance_x(): .2f} m.')
                    ball.set_velocity_y(0)
                    ball.set_distance_y(radius)
                    current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                    values_array.append(current_values)
                    #time_after_collision.append(time_array[i])
                    ball.set_time_after_collision(time_array[i])
        
                else:
                    print('An Error has occured, regarding collision')
            elif ball.get_motion_status() == 'sliding':
                if i == 0:
                    #time_after_collision.append(0) #Če se gibanje začne z drsnim trenjem
                    ball.set_time_after_collision(0)
                    
                if ball.get_velocity_x()>0:
                    
                    if abs(ball.get_velocity_x()) - abs(ball.get_velocity_phi()) * ball.get_radius() < 0.1:
                        print(f'Time elapsed after sliding is{i*a/b: .2f} s.')
                        print(f'Distance x after sliding is{ball.get_distance_x(): .2f} m.')
                        print(f'Distance phi after sliding is{ball.get_distance_phi(): .2f} m.')
                        ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
                        
                        #time_after_sliding.append(time_array[i])
                        #distance_phi_after_sliding.append(ball.get_distance_phi())
                        #velocity_x_after_sliding.append(ball.get_velocity_x())
                        #distance_x_after_sliding.append(ball.get_distance_x())
                        """Used to animate ball movement:"""
                        ball.set_time_after_sliding(time_array[i])
                        ball.set_distance_phi_after_sliding(ball.get_distance_phi())
                        ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                        ball.set_distance_x_after_sliding(ball.get_distance_x())
                        print(f'Velocity_phi after sliding:{ball.get_velocity_phi(): .2f}rad/s')

            
            
                    else:
                        y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_phi(), ball.get_velocity_phi()])
                        solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)  
                        ball.set_distance_x(solution.y[0][-1])
                        ball.set_velocity_x(solution.y[1][-1])
                        ball.set_distance_phi(solution.y[2][-1])
                        ball.set_velocity_phi(solution.y[3][-1])
                        

                        
                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
                
                elif ball.get_velocity_x()<0:
                    """If the ball is moving in negative x direction."""
                    if abs(ball.get_velocity_x()) - abs(ball.get_velocity_phi()) * ball.get_radius() < 0.1:
                        print(f'Time elapsed after sliding is{i*a/b: .2f} s.')
                        print(f'Distance x after sliding is{ball.get_distance_x(): .2f} m.')
                        print(f'Distance phi after sliding is{ball.get_distance_phi(): .2f} rad.')
                        ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
                        
                        #time_after_sliding.append(time_array[i])
                        #distance_phi_after_sliding.append(ball.get_distance_phi())
                        #velocity_x_after_sliding.append(ball.get_velocity_x())
                        #distance_x_after_sliding.append(ball.get_distance_x())
                        """Used to animate ball movement:"""
                        ball.set_time_after_sliding(time_array[i])
                        ball.set_distance_phi_after_sliding(ball.get_distance_phi())
                        ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                        ball.set_distance_x_after_sliding(ball.get_distance_x())
                        print(f'Velocity_phi after sliding:{ball.get_velocity_phi(): .2f}rad/s')

            
            
                    else:
                        y_0 = np.array([abs(ball.get_distance_x()), abs(ball.get_velocity_x()), abs(ball.get_distance_phi()), abs(ball.get_velocity_phi())])
                        solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)  
                        ball.set_distance_x(solution.y[0][-1]*(-1))
                        ball.set_velocity_x(solution.y[1][-1]*(-1))
                        ball.set_distance_phi(solution.y[2][-1]*(-1))
                        ball.set_velocity_phi(solution.y[3][-1]*(-1))

                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
                    
                    
                    
            
        
            elif ball.get_motion_status() == 'rolling':
                if i == 0:
                    #time_after_sliding.append(0) #če ni drsenja(ni še upoštevano, če je kotaljenje po poševnem metu!)
                    #velocity_x_after_sliding.append(ball.get_velocity_x())
                    #distance_x_after_sliding.append(ball.get_distance_x())
                    
                    ball.set_time_after_sliding(0) #če ni drsenja(ni še upoštevano, če je kotaljenje po poševnem metu!)
                    ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                    ball.set_distance_x_after_sliding(ball.get_distance_x())
                    
                if ball.get_distance_x()>0:
                    if abs(ball.get_velocity_x()) < 0.1:
        
                        ball.set_velocity_x(0)
                        ball.set_velocity_phi(0)
                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
                        
                        #distance_phi_after_rolling.append(ball.get_distance_phi())
                        #time_after_rolling.append(time_array[i])
                        
                        ball.set_distance_phi_after_rolling(ball.get_distance_phi())
                        ball.set_time_after_rolling(time_array[i])
                        
                            
            
                        print(f'Time elapsed after rolling is{i*a/b: .2f} s.')
                        print(f'Distance x after rolling is{ball.get_distance_x(): .2f} m.')
                        print(f'Distance_phi after rolling: {ball.get_distance_phi()} rad.')
                        
                    else:
                        y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x()])
                        solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                        ball.set_distance_x(solution.y[0][-1])
                        ball.set_velocity_x(solution.y[1][-1])
                        ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                        #ball.set_distance_phi(distance_phi_after_sliding[0][0]+(ball.get_distance_x()-distance_x_after_sliding[0])/ball.get_radius())
                        ball.set_distance_phi(ball.get_distance_phi_after_sliding()+(ball.get_distance_x()-ball.get_distance_x_after_sliding())/ball.get_radius())
                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
                elif ball.get_velocity_x()<0:
                    if abs(ball.get_velocity_x()) < 0.1:
        
                        ball.set_velocity_x(0)
                        ball.set_velocity_phi(0)
                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
                        
                        #distance_phi_after_rolling.append(ball.get_distance_phi())
                        #time_after_rolling.append(time_array[i])
                        
                        ball.set_distance_phi_after_rolling(ball.get_distance_phi())
                        ball.set_time_after_rolling(time_array[i])

                            
            
                        print(f'Time elapsed after rolling is{i*a/b: .2f} s.')
                        print(f'Distance x after rolling is{ball.get_distance_x(): .2f} m.')
                        print(f'Distance_phi after rolling: {ball.get_distance_phi()} rad.')
                        
                    else:
                        y_0 = np.array([abs(ball.get_distance_x()), abs(ball.get_velocity_x())])
                        solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                        ball.set_distance_x(solution.y[0][-1]*(-1))
                        ball.set_velocity_x(solution.y[1][-1]*(-1))
                        ball.set_velocity_phi(ball.get_velocity_x()*(-1) / ball.get_radius())
                        #ball.set_distance_phi(distance_phi_after_sliding[0][0]+(ball.get_distance_x()-distance_x_after_sliding[0])/ball.get_radius())
                        ball.set_distance_phi(ball.get_distance_phi_after_sliding()+(ball.get_distance_x()-ball.get_distance_x_after_sliding())/ball.get_radius())
                        current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                        values_array.append(current_values)
            
            elif ball.get_motion_status() == 'not_moving':
                current_values=[ball.get_distance_x(),ball.get_velocity_x(),ball.get_distance_y(),ball.get_velocity_y(),ball.get_distance_phi(),ball.get_velocity_phi()]
                values_array.append(current_values)
            else:
                print('Error')
                    
                
                
        
    return values_array #values_array treba spremenit v 3-D


if __name__ == '__main__':
    tic=time.time()
    H = 1.5
    m = 0.5
    v_0 = 10
    g = 9.81
    alpha_0 = 30 * np.pi / 180
    x0 = 0
    phi0 = 0
    v_phi0 = 0
    radius = 0.10
    epsilon = 0.2
    mi = 0.3
    f = 0.2
    theta = f / radius
    a=20
    b=10000
    time_array = np.linspace(0, a, b)
    t = time_array
    #distance_x = 0
    #distance_y = H
    #distance_phi = 0
    #velocity_x = v_0 * np.cos(alpha_0)
    #velocity_y = v_0 * np.sin(alpha_0)
    #velocity_phi = 0
    
    
    b1 = Ball(1, 0, v_0*np.cos(alpha_0), H,v_0 * np.sin(alpha_0), 0, 0, radius,1)
    #b1 = Ball(1, 300, v_0*np.cos(alpha_0), H,v_0 * np.sin(alpha_0), 0, 0, radius)
    b2 = Ball(2, 0, v_0*np.cos(alpha_0), H,2*v_0 * np.sin(alpha_0), 0, 0, radius,1)
    #def __init__(self, ID, distance_x, velocity_x,distance_y,velocity_y, distance_phi, velocity_phi,radius,mass):
    balls = [b2]
    
    world1 = World(2)
    world1.add_ball(b1)

    
    
    distance_x_array=[b1.get_distance_x()]
    velocity_x_array = [b1.get_velocity_x()]
    distance_y_array = [b1.get_distance_y()]
    velocity_y_array = [b1.get_velocity_y()]
    distance_phi_array=[b1.get_distance_phi()]
    velocity_phi_array = [b1.get_velocity_phi()]
    
    
    #rez1=time_solution(b1)
    rez1=time_solution(b1)

    for i in range(len(rez1)):
        distance_x_array.append(rez1[i][0])
        velocity_x_array.append(rez1[i][1])
        distance_y_array.append(rez1[i][2])
        velocity_y_array.append(rez1[i][3])
        distance_phi_array.append(rez1[i][4])
        velocity_phi_array.append(rez1[i][5])
    plt.plot(time_array, velocity_x_array, label='Velocity_x')
    plt.show()
    #plt.plot(time_array, distance_x_array, label='Distance_x')
    #plt.show()
    plt.plot(time_array, velocity_y_array, label='Velocity_y')
    plt.show()
    plt.plot(time_array, distance_y_array, label='Distance_y')
    plt.show()
    """plt.plot(distance_x_array, distance_y_array, label='Center of mass location')
    plt.show()"""
    #plt.plot(time_array, velocity_phi_array, label='Vlocity_phi')
    #plt.show()
    plt.legend();
    plt.xlabel('t[s]');
    plt.ylabel('vx,vy[m/s], y[m]');
    plt.title('Ball movement');
    plt.xlim([0, np.max(b1.get_time_after_rolling())+3])
    
    
    
    #ANIMATION
    fig = plt.figure()
    ax = plt.axes(xlim=(-1.5+ distance_x_array[0],distance_x_array[-1]+10),
                  ylim=(0,np.amax(distance_y_array)+2))

    #lines = plt.plot([], "o")
    #line = lines[0]
    #line2 = lines[0]
    
    line, = ax.plot([], [], 'o', lw=2, color='b')
    line2, = ax.plot([], [], '.', lw=2, color='r')
    
    #ball = plt.Circle((30,5), r*10, fc='y', fill=False)
   # ax.add_patch(ball)
    title = ax.text(0.5,1.05, "Animation of ball movement", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

    
    #other setup
    #plt.axis("scaled")
    #plt.xlim(-1.5,330)
    #plt.ylim(0,17)
    
    time_template = 'time = %.1fs'
    velocity_template1 = 'velocity_x = %.1fm/s'
    velocity_template2 = 'velocity_y = %.1fm/s'
    velocity_template3 = 'velocity_phi = %.1frad/s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    velocity_text1 = ax.text(0.05, 0.85, '', transform=ax.transAxes)
    velocity_text2 = ax.text(0.05, 0.80, '', transform=ax.transAxes)
    velocity_text3 = ax.text(0.05, 0.75, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        #update plot
        x_out = [distance_x_array[frame],distance_x_array[frame]+np.sin(distance_phi_array[frame])*radius]
        y_out = [distance_y_array[frame], distance_y_array[frame]+np.cos(distance_phi_array[frame])*radius]
        x_out2 = distance_x_array[frame]+10
        vx_out = velocity_x_array[frame]
        vy_out = velocity_y_array[frame]
        vphi_out = velocity_phi_array[frame]
        line.set_data((x_out, y_out))
        line2.set_data((x_out2, y_out))
        #ball.center= ((x_out, y_out))
        
        time_text.set_text(time_template%(frame*a/b))
        velocity_text1.set_text(velocity_template1%(vx_out))
        velocity_text2.set_text(velocity_template2%(vy_out))
        velocity_text3.set_text(velocity_template3%(vphi_out))
        return line, velocity_text1, line2


    
    anim = FuncAnimation(fig, animate, frames = 100, interval = a/b*1000)
    plt.grid()
    plt.show()
    
    
        

    
    
    
    
    tok=time.time()
    computing_time=tok-tic
    print(f'Computing time = {computing_time: .2f}s.')
    print(f'Motion status of ball 1 = {b1.get_motion_status()}')
    print(f'Motion status of ball 2 = {b2.get_motion_status()}')

    
