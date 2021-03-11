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
    return y[1], -mi * g, y[3], (5 * mi * g) / (2 * r)


def fun_rolling(t, y):
    """
    Function is used to solve differential equations,
    while the object is rolling.
    """
    return y[1], -5 / 7 * g * theta


def get_state(distance_x, velocity_x,distance_y,velocity_y, distance_phi, velocity_phi,r):
    """
    Function used to determine motion state of the object.
    """
    if velocity_x == 0 and distance_y == r:
        state = "not_moving"
    elif distance_y > r or velocity_y > 0:
        state = "projectile_motion"
    elif distance_y < r and abs(velocity_y) > 0:
        state = "collision"
    elif velocity_y == 0 and distance_y == r and velocity_x - velocity_phi * r > 0:
        state = "sliding"
    elif velocity_y == 0 and distance_y == r and velocity_x - velocity_phi * r < 0.001 and velocity_x > 0:
        state = "rolling"
    else:
        print('Error at determinating motion state!')

    return state





def time_solution(distance_x, distance_y, distance_phi, velocity_x, velocity_y, velocity_phi,r):
    values_array = []
    int = -1
    state=get_state(distance_x, velocity_x, distance_y, velocity_y, distance_phi, velocity_phi,r)
    
    for i in range(len(time_array) - 1):
        if i!=0:
            state = get_state(values_array[int][0],values_array[int][1],values_array[int][2],values_array[int][3],values_array[int][4],values_array[int][5],r)
        if state == 'projectile_motion':
            if i==0:
                y_0 = np.array([distance_x, velocity_x, distance_y, velocity_y])
            else:
                y_0 = np.array([values_array[int][0], values_array[int][1], values_array[int][2], values_array[int][3]])
            solution = solve_ivp(fun_projectile_motion, (time_array[i], time_array[i + 1]), y_0)
            distance_x = solution.y[0][-1]
            velocity_x = solution.y[1][-1]
            distance_y = solution.y[2][-1]
            velocity_y = solution.y[3][-1]
            current_values=[distance_x,velocity_x,distance_y,velocity_y,0,0]
            values_array.append(current_values)
            int += 1

        elif state == 'collision':
            previous_velocity_y = values_array[int][3]
            if abs(previous_velocity_y * epsilon) >= 0.1:
                previous_velocity_x = values_array[int][1]
                previous_distance_x = values_array[int][0]
                velocity_y = previous_velocity_y * epsilon * (-1)
                distance_y = r
                current_values = [previous_distance_x, previous_velocity_x, distance_y, velocity_y, 0, 0]
                values_array.append(current_values)
                int += 1
            elif abs(previous_velocity_y * epsilon) < 0.1:
                print(f'Time elapsed after last collision is{i*a/b: .2f} s.')
                print(f'Distance x after last collision is{previous_distance_x: .2f} m.')
                velocity_y = 0
                distance_y = r
                current_values = [previous_distance_x, previous_velocity_x, distance_y, velocity_y, 0, 0]
                values_array.append(current_values)
                int += 1
            else:
                print('An Error has occured, regarding collision')
        elif state == 'sliding':
            if i!=0:
                previous_distance_phi = values_array[int][4]
                previous_distance_x = values_array[int][0]
                previous_velocity_x = values_array[int][1]
                previous_velocity_phi = values_array[int][5]
            else:
                previous_distance_phi = distance_phi
                previous_distance_x = distance_x
                previous_velocity_x = velocity_x
                previous_velocity_phi = velocity_phi
            if previous_velocity_x - previous_velocity_phi * r < 0.01:
                print(f'Time elapsed after sliding is{i*a/b: .2f} s.')
                print(f'Distance x after sliding is{previous_distance_x: .2f} m.')
                velocity_phi = previous_velocity_x / r
                current_values = [previous_distance_x, previous_velocity_x, r, 0, previous_distance_phi, velocity_phi]
                values_array.append(current_values)
                int += 1

            else:
                if i!=0:
                    previous_distance_phi = values_array[int][4]
                    previous_distance_x = values_array[int][0]
                    previous_velocity_x = values_array[int][1]
                    previous_velocity_phi = values_array[int][5]
                else:
                    previous_distance_phi = distance_phi
                    previous_distance_x = distance_x
                    previous_velocity_x = velocity_x
                    previous_velocity_phi = velocity_phi
                y_0 = np.array([previous_distance_x, previous_velocity_x, previous_distance_phi, previous_velocity_phi])
                solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)
                distance_x = solution.y[0][-1]
                velocity_x = solution.y[1][-1]
                distance_phi = solution.y[2][-1]
                velocity_phi = solution.y[3][-1]
                current_values = [distance_x, velocity_x,r,0, distance_phi, velocity_phi]
                values_array.append(current_values)
                int+=1
        elif state == 'rolling':
            previous_velocity_x = values_array[int][1]
            if previous_velocity_x < 0.01:

                q = values_array[int][0]
                q2 = 0
                q3 = values_array[int][2]
                q4 = values_array[int][3]
                q5 = values_array[int][4]
                q6 = values_array[int][5]
                current_values = [q, q2, q3, q4, q5, q6]
                values_array.append(current_values)
                int+=1
                print(f'Time elapsed after rolling is{i*a/b: .2f} s.')
                print(f'Distance x after rolling is{previous_distance_x: .2f} m.')

            else:
                previous_distance_x = values_array[int][0]
                previous_velocity_x = values_array[int][1]
                y_0 = np.array([previous_distance_x, previous_velocity_x])
                solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                distance_x = solution.y[0][-1]
                velocity_x = solution.y[1][-1]
                velocity_phi = previous_velocity_x / r
                current_values = [distance_x, velocity_x,r,0,0,velocity_phi]
                values_array.append(current_values)
                int+=1

        elif state == 'not_moving':
            q = values_array[int][0]
            q2 = values_array[int][1]
            q3 = values_array[int][2]
            q4 = values_array[int][3]
            q5 = values_array[int][4]
            q6 = values_array[int][5]
            current_values = [q, q2, q3, q4, q5, q6]
            values_array.append(current_values)
            int+=1
        else:
            print('Error')

    return values_array



if __name__ == '__main__':
    H = 1.5
    m = 0.5
    v_0 = 30
    g = 9.81
    alpha_0 = 30 * np.pi / 180
    r = 0.15
    epsilon = 0.2
    mi = 0.1
    f = 0.08
    theta = f / r
    a=20
    b=10000
    time_array = np.linspace(0, a, b)
    t = time_array
    distance_x = 0
    distance_y = H
    distance_phi = 0
    velocity_x = v_0 * np.cos(alpha_0)
    velocity_y = v_0 * np.sin(alpha_0)
    velocity_phi = 0
    rez=time_solution(distance_x, distance_y, distance_phi, velocity_x, velocity_y, velocity_phi,r)
    distance_x_array=[0]
    velocity_x_array = [velocity_x]
    distance_y_array = [H]
    velocity_y_array = [velocity_y]
    velocity_phi_array = [0]
    for i in range(len(rez)):
        distance_x_array.append(rez[i][0])
        velocity_x_array.append(rez[i][1])
        distance_y_array.append(rez[i][2])
        velocity_y_array.append(rez[i][3])
        velocity_phi_array.append(rez[5])
    plt.plot(time_array, velocity_x_array, label='Velocity_x')
    plt.show()
    #plt.plot(time_array, distance_x_array, label='Distance_x')
    #plt.show()
    plt.plot(time_array, velocity_y_array, label='Velocity_y')
    plt.show()
    plt.plot(time_array, distance_y_array, label='Distance_y')
    plt.show()
    #plt.plot(distance_x_array, distance_y_array,, label='Center of mass location')
    #plt.show()
    #plt.plot(time_array, velocity_phi_array, label='Vlocity_phi')
    #plt.show()
    plt.legend();
    plt.xlabel('t[s]');
    plt.ylabel('vx,vy[m/s], y[m]');
    plt.title('Ball movement');



