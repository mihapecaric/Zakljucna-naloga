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


def get_state(distance_x, velocity_x,distance_y,velocity_y, distance_phi, velocity_phi):
    """
    Function used to determine motion state of the object.
    """
    if velocity_x==0:
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
        state="not_moving"

    return state





def time_solution(distance_x, distance_y, distance_phi, velocity_x, velocity_y, velocity_phi,r):
    abc = []
    stevec=-1
    state=get_state(distance_x, velocity_x, distance_y, velocity_y, distance_phi, velocity_phi)
    #print(state)

    for i in range(len(time_array) - 1):
        #print(i)
        #print(stevec)
        #print(abc)
        if i!=0:
            state = get_state(abc[stevec][0],abc[stevec][1],abc[stevec][2],abc[stevec][3],abc[stevec][4],abc[stevec][5])
            #print(state)


        if state == 'projectile_motion':
            #print(abc)

            if i==0:
                y_0 = np.array([distance_x, velocity_x, distance_y, velocity_y])
            else:
                y_0 = np.array([abc[stevec][0], abc[stevec][1], abc[stevec][2], abc[stevec][3]])
            solution = solve_ivp(fun_projectile_motion, (time_array[i], time_array[i + 1]), y_0)
            distance_x = solution.y[0][-1]
            velocity_x = solution.y[1][-1]
            distance_y = solution.y[2][-1]
            velocity_y = solution.y[3][-1]
            #distance velocity  NAJPREJ X POL Y,,,,,KOT, KOTNA HITROST
            a=[distance_x,velocity_x,distance_y,velocity_y,0,0]
            abc.append(a)
            stevec+=1



        elif state == 'collision':

            prejsni_velocity_y = abc[stevec][3]
            if abs(prejsni_velocity_y * epsilon) >= 0.1:
                prejsni_velocity_x = abc[stevec][1]
                prejsni_distance_x = abc[stevec][0]
                velocity_y = prejsni_velocity_y * epsilon * (-1)
                distance_y = r
                a = [prejsni_distance_x, prejsni_velocity_x, distance_y, velocity_y, 0, 0]
                abc.append(a)
                stevec += 1
            elif abs(prejsni_velocity_y * epsilon) < 0.1:
                velocity_y = 0
                distance_y = r
                a = [prejsni_distance_x, prejsni_velocity_x, distance_y, velocity_y, 0, 0]
                abc.append(a)
                stevec += 1
            else:
                print('An Error has occured, regarding collision')
            #print(a)
            #break;
        elif state == 'sliding':
            if i!=0:
                prejsni_distance_phi = abc[stevec][4]
                prejsni_distance_x = abc[stevec][0]
                prejsni_velocity_x = abc[stevec][1]
                prejsni_velocity_phi = abc[stevec][5]
            else:
                prejsni_distance_phi = distance_phi
                prejsni_distance_x = distance_x
                prejsni_velocity_x = velocity_x
                prejsni_velocity_phi = velocity_phi
            if prejsni_velocity_x - prejsni_velocity_phi * r < 0.01:

                velocity_phi = prejsni_velocity_x / r
                ###################################
                a = [prejsni_distance_x, prejsni_velocity_x, r, 0, prejsni_distance_phi, velocity_phi]
                abc.append(a)
                stevec += 1

            else:
                if i!=0:
                    prejsni_distance_phi = abc[stevec][4]
                    prejsni_distance_x = abc[stevec][0]
                    prejsni_velocity_x = abc[stevec][1]
                    prejsni_velocity_phi = abc[stevec][5]
                else:
                    prejsni_distance_phi = distance_phi
                    prejsni_distance_x = distance_x
                    prejsni_velocity_x = velocity_x
                    prejsni_velocity_phi = velocity_phi
                y_0 = np.array([prejsni_distance_x, prejsni_velocity_x, prejsni_distance_phi, prejsni_velocity_phi])
                solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)
                distance_x = solution.y[0][-1]
                velocity_x = solution.y[1][-1]
                distance_phi = solution.y[2][-1]
                velocity_phi = solution.y[3][-1]
                a = [distance_x, velocity_x,r,0, distance_phi, velocity_phi]
                abc.append(a)
                stevec+=1
        elif state == 'rolling':
            prejsni_velocity_x = abc[stevec][1]
            if prejsni_velocity_x < 0.01:

                q = abc[stevec][0]
                q2 = 0
                q3 = abc[stevec][2]
                q4 = abc[stevec][3]
                q5 = abc[stevec][4]
                q6 = abc[stevec][5]
                a = [q, q2, q3, q4, q5, q6]
                abc.append(a)
                stevec+=1
                print(((i)*20)/10000)
            else:
                prejsni_distance_x = abc[stevec][0]
                prejsni_velocity_x = abc[stevec][1]
                y_0 = np.array([prejsni_distance_x, prejsni_velocity_x])
                solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                distance_x = solution.y[0][-1]
                velocity_x = solution.y[1][-1]
                velocity_phi = prejsni_velocity_x / r
                a = [distance_x, velocity_x,r,0,0,velocity_phi]
                abc.append(a)
                stevec+=1
        elif state == 'not_moving':

            #velocity_x = 0
            #velocity_phi = 0
            q = abc[stevec][0]
            q2 = abc[stevec][1]
            q3 = abc[stevec][2]
            q4 = abc[stevec][3]
            q5 = abc[stevec][4]
            q6 = abc[stevec][5]
            a = [q, q2, q3, q4, q5, q6]
            abc.append(a)
            stevec+=1
        else:
            print('Error')

        """
        distance_x_array = np.append(distance_x_array, distance_x)
        velocity_x_array = np.append(velocity_x_array, velocity_x)
        distance_y_array = np.append(distance_y_array, distance_y)
        velocity_y_array = np.append(velocity_y_array, velocity_y)
        distance_phi_array = np.append(distance_phi_array, distance_phi)
        velocity_phi_array = np.append(velocity_phi_array, velocity_phi)
        """

    return abc



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
    time_array = np.linspace(0, 20, 10000)
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
    distance_y_array = [0]
    velocity_y_array = [velocity_y]
    #print(rez)
    for i in range(len(rez)):
        distance_x_array.append(rez[i][0])
        velocity_x_array.append(rez[i][1])
        distance_y_array.append(rez[i][2])
        velocity_y_array.append(rez[i][3])
    print(len(distance_x_array))
    plt.plot(time_array, velocity_x_array)
    plt.show()
    #plt.plot(time_array, distance_x_array)
    #plt.show()
    plt.plot(time_array, velocity_y_array)
    plt.show()
    plt.plot(time_array, distance_y_array)
    plt.show()
    #plt.plot(distance_x_array, distance_y_array)
    #plt.show()


