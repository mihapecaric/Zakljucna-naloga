import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing()
from scipy.integrate import solve_ivp


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
    return y[1], -mi*g, y[3], (5*mi*g)/(2*r) 

def fun_rolling(t, y):
    """
    Function is used to solve differential equations,
    while the object is rolling.
    """
    return y[1], -5/7*g*theta


def get_state(distance_x,distance_y,distance_phi,velocity_x,velocity_y,velocity_phi):
    """
    Function used to determine motion state of the object.
    """
    if distance_y > r or velocity_y > 0:
        state = "projectile_motion"
    elif distance_y < r and abs(velocity_y) > 0:
        state = "collision"
    elif velocity_y == 0 and distance_y == r and velocity_x - velocity_phi * r > 0:
        state = "sliding"
    elif velocity_y == 0 and distance_y == r and velocity_x - velocity_phi * r == 0 and velocity_x > 0:
        state = "rolling"
    else:
        state = "not_moving"
    return state


def time_solution(distance_x, distance_y, distance_phi, velocity_x, velocity_y, velocity_phi):
    """
    Solves differential equations for t in time_array
    """
    
    distance_x_array = np.array([])
    velocity_x_array = np.array([])
    distance_y_array = np.array([])
    velocity_y_array = np.array([])
    distance_phi_array = np.array([])
    velocity_phi_array = np.array([])

    for i in range(len(time_array)-1):
        state=get_state(distance_x,distance_y,distance_phi,velocity_x,velocity_y,velocity_phi)
        if state == 'projectile_motion':
            y_0 = np.array([distance_x, velocity_x, distance_y, velocity_y])
            solution = solve_ivp(fun_projectile_motion, (time_array[i], time_array[i+1]), y_0)
            distance_x = solution.y[0][-1]
            velocity_x = solution.y[1][-1]
            distance_y = solution.y[2][-1]
            velocity_y = solution.y[3][-1]

        elif state == 'collision':
            if abs(velocity_y * epsilon) >= 0.1:
                velocity_y == velocity_y * epsilon * (-1)
                distance_y == r
            elif abs(velocity_y * epsilon) < 0.1:
                velocity_y == 0
                distance_y == r
            else:
                print('An Error has occured, regarding collision')
        elif state == 'sliding':
            if velocity_x - velocity_phi * r < 0.01:
                velocity_phi = velocity_x / r
            else:
                y_0 = np.array([distance_x, velocity_x, distance_phi, velocity_phi])
                solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i+1]), y_0)
                distance_x = solution.y[0][-1]
                velocity_x = solution.y[1][-1]
                distance_phi = solution.y[2][-1]
                velocity_phi = solution.y[3][-1]
        elif state == 'rolling':
            if velocity_x < 0.01:
                velocity_x = 0
            else:
                y_0 = np.array([distance_x, velocity_x])
                solution = solve_ivp(fun_rolling, (time_array[i], time_array[i+1]), y_0)
                distance_x = solution.y[0][-1]
                velocity_x = solution.y[1][-1]
                velocity_phi = velocity_x / r
        elif state == 'not moving':
            velocity_x = 0
            velocity_phi = 0
        else:
            print('Error')
        distance_x_array = np.append(distance_x_array, distance_x)
        velocity_x_array = np.append(velocity_x_array, velocity_x)
        distance_y_array = np.append(distance_y_array, distance_y)
        velocity_y_array = np.append(velocity_y_array, velocity_y)
        distance_phi_array = np.append(distance_phi_array, distance_phi)
        velocity_phi_array = np.append(velocity_phi_array, velocity_phi)
if __name__ == '__main__':
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
    time_array=np.linspace(0, 20, 10000)
    t=time_array
    distance_x = 0
    distance_y = H
    distance_phi = 0
    velocity_x = v_0*np.cos(alpha_0)
    velocity_y = v_0*np.sin(alpha_0)
    velocity_phi = 0
    time_solution(distance_x, distance_y, distance_phi, velocity_x, velocity_y, velocity_phi)
    plt.plot(time_array, distance_x_array)
    