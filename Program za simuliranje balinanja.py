import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

class Balls:
    def __init__(self, list, time_array):
        self.time_array = time_array
        self.list = []
        for i in list:
            self.add_ball(i)

    def add_ball(self, ball):
        self.list.append(ball)

    def print(self):
        for i in self.list:
            i.print()

    def check_collision(self):
        """Function that determinates whether 2 objects are coliding or not. It is
        also used to prevent objects from getting stuck"""
        bool1 = False
        seznam = []
        for i in self.list:
            for j in self.list:
                id1 = i.ID
                id2 = j.ID
                if id1 == id2: #used to skip same ball in list
                    continue
                x1 = i.distance[0]
                y1 = i.distance[1]
                z1 = i.distance[2]
                rad1 = i.radius
                x2 = j.distance[0]
                y2 = j.distance[1]
                z2 = j.distance[2]
                rad2 = j.radius
                if i.coll == 1 and j.coll == 1:
                    if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) < rad1 + rad2:
                        i.coll = 0 #used to prevent balls from getting stuck
                        j.coll = 0
                        i.num_colls = i.num_colls + 1
                        j.num_colls = j.num_colls + 1
                        seznam.append([i, j])
                        bool1=True
                        break
                else:
                    i.coll = 1 #enables balls to collide again
                    j.coll = 1
        return seznam, bool1

    def time_solution(self, a, b):
        """Function that computes motion parameters for each ball, at each time interval"""
        for i in range(len(self.time_array) - 1):
            for ball in self.list:

                theta = f/ball.radius
                def fun_projectile_motion(t, y):
                    """
                    Function is used to solve differential equations,
                    while the object is in the air.
                    """
                    return y[1], 0, y[3], 0, y[5], - g

                def fun_slipping_rolling(t, y):
                    """
                    Function is used to solve differential equations,
                    while the object is slipping and rolling.
                    """
                    return y[1], -mi * g, y[3], (5 * mi * g) / (2 * ball.radius)

                def fun_slipping_rolling_backwards(t, y):
                    """
                    Function is used to solve differential equations,
                    while the object is slipping and rolling in negative x or y direction.
                    """
                    return y[1], mi * g, y[3], (5 * -mi * g) / (2 * ball.radius)

                def fun_rolling(t, y):
                    """
                    Function is used to solve differential equations,
                    while the object is rolling.
                    """
                    return y[1], -5 / 7 * g * theta

                def fun_rolling_backwards(t, y):
                    """
                    Function is used to solve differential equations,
                    while the object is rolling in negative x or y direction.
                    """
                    return y[1], 5 / 7 * g * theta

                """BOULES COURT:"""
                if ball.distance[0]>15-ball.radius:
                    ball.velocity[0] = -ball.velocity[0]
                    ball.velocity[3] = -ball.velocity[3]
                if ball.distance[0]<0:
                    ball.velocity[0] = -ball.velocity[0]
                    ball.velocity[3] = -ball.velocity[3]
                if ball.distance[1]>2-ball.radius:
                    ball.velocity[1] = -ball.velocity[1]
                    ball.velocity[4] = -ball.velocity[4]
                if ball.distance[1]<-2+ball.radius:
                    ball.velocity[1] = -ball.velocity[1]
                    ball.velocity[4] = -ball.velocity[4]
                if abs(ball.velocity[0] >= ball.velocity[1]):

                    if ball.velocity[1] != 0:
                        ratio = ball.velocity[0] / ball.velocity[1]


                    if i == 0:
                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)
                    if ball.get_motion_status() == 'projectile_motion':

                        y_0 = np.array(
                            [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1], ball.distance[2],
                             ball.velocity[2]])
                        solution = solve_ivp(fun_projectile_motion, (time_array[i], time_array[i + 1]), y_0)

                        ball.distance[0] = solution.y[0][-1]
                        ball.velocity[0] = solution.y[1][-1]
                        ball.distance[1] = solution.y[2][-1]
                        ball.velocity[1] = solution.y[3][-1]
                        ball.distance[2] = solution.y[4][-1]
                        ball.velocity[2] = solution.y[5][-1]

                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)


                    elif ball.get_motion_status() == 'collision':
                        if abs(ball.velocity[2] * epsilon) >= 0.1:
                            ball.velocity[2] = ball.velocity[2] * epsilon * (-1)
                            current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                              ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                              ball.distance[4], ball.velocity[4]]
                            ball.list.append(current_values)

                        elif abs(ball.velocity[2] * epsilon) < 0.1:
                            ball.velocity[2] = 0
                            ball.distance[2] = ball.radius
                            current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                              ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                              ball.distance[4], ball.velocity[4]]
                            ball.list.append(current_values)
                            ball.time_after_collision = time_array[i]

                        else:
                            print('An Error has occured, regarding collision')
                    elif ball.get_motion_status() == 'sliding_x':
                        if ball.velocity[0] > 0:
                            if abs(ball.velocity[0]) - abs(ball.velocity[3]) * ball.radius < 0.1:
                                ball.velocity[3] = ball.velocity[0] / ball.radius
                                ball.velocity[4] = ball.velocity[1] / ball.radius
                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                                """Used to animate ball movement:"""
                                ball.time_after_sliding_x = time_array[i]
                                ball.distance_after_sliding[3] = ball.distance[3]
                                ball.velocity_after_sliding[0] = ball.velocity[0]
                                ball.distance_after_sliding[0] = ball.distance[0]

                                ball.distance_after_sliding[4] = ball.distance[4]#NA NOVO!!!!
                                ball.velocity_after_sliding[1] = ball.velocity[1]
                                ball.distance_after_sliding[1] = ball.distance[1]




                            else:
                                y_0 = np.array([ball.distance[0], ball.velocity[0], ball.distance[3], ball.velocity[3]])
                                solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)
                                ball.distance[0] = solution.y[0][-1]
                                ball.velocity[0] = solution.y[1][-1]
                                ball.distance[3] = solution.y[2][-1]
                                ball.velocity[3] = solution.y[3][-1]

                                if ball.velocity[1] == 0: # used to prevent division by 0 (if v_y=0->ratio=0)
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]
                                    ball.velocity[1] = 0
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = 0
                                else:
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]
                                    ball.velocity[1] = ball.velocity[0] / ratio
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = ball.velocity[3] / ratio


                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                        else:
                            if abs(ball.velocity[0]) - abs(ball.velocity[3]) * ball.radius < 0.1:
                                ball.velocity[3] = ball.velocity[0] / ball.radius
                                ball.velocity[4] = ball.velocity[1] / ball.radius  # NA NOVO!!!!
                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                                """Used to animate ball movement:"""
                                ball.time_after_sliding = time_array[i]
                                ball.distance_after_sliding[3] = ball.distance[3]
                                ball.velocity_after_sliding[0] = ball.velocity[0]
                                ball.distance_after_sliding[0] = ball.distance[0]

                                ball.distance_after_sliding[4] = ball.distance[4]  # NA NOVO!!!!
                                ball.velocity_after_sliding[1] = ball.velocity[1]
                                ball.distance_after_sliding[1] = ball.distance[1]



                            else:
                                y_0 = np.array([ball.distance[0], ball.velocity[0], ball.distance[3], ball.velocity[3]])
                                solution = solve_ivp(fun_slipping_rolling_backwards, (time_array[i], time_array[i + 1]),
                                                     y_0)
                                ball.distance[0] = solution.y[0][-1]
                                ball.velocity[0] = solution.y[1][-1]
                                ball.distance[3] = solution.y[2][-1]
                                ball.velocity[3] = solution.y[3][-1]

                                if ball.velocity[1] == 0:
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]  # NA NOVO!!!!
                                    ball.velocity[1] = 0
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = 0
                                else:
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]  # NA NOVO!!!!
                                    ball.velocity[1] = ball.velocity[0] / ratio
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = ball.velocity[3] / ratio

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                    elif ball.get_motion_status() == 'rolling_x':

                        if i == 0:
                            # ball.set_time_after_sliding(0)
                            ball.velocity_after_sliding[0] = ball.velocity[0]
                            ball.distance_after_sliding[0] = ball.distance[0]
                            ball.distance_after_sliding[3] = ball.distance[3]

                            ball.velocity_after_sliding[1] = ball.velocity[1]#NA NOVO!!!!!
                            ball.distance_after_sliding[1] = ball.distance[1]
                            ball.distance_after_sliding[4] = ball.distance[4]

                        if ball.velocity[0] > 0:
                            if abs(ball.velocity[0]) < 0.1:

                                ball.velocity[0] = 0
                                ball.velocity[3] = 0

                                ball.velocity[1] = 0 #NA NOVO!!!!
                                ball.velocity[4] = 0

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                                ball.distance_after_rolling[3] = ball.distance[3]
                                ball.distance_after_rolling[4] = ball.distance[4]#NA NOVO!!!
                                ball.time_after_rolling_x = time_array[i]


                            else:
                                y_0 = np.array([ball.distance[0], ball.velocity[0]])
                                solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                                ball.distance[0] = solution.y[0][-1]
                                ball.velocity[0] = solution.y[1][-1]
                                ball.velocity[3] = ball.velocity[0] / ball.radius
                                ball.distance[3] = ball.distance_after_sliding[3] + (
                                        ball.distance[0] - ball.distance_after_sliding[0]) / ball.radius
                                if ball.velocity[1] == 0:
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]  # NA NOVO!!!!
                                    ball.velocity[1] = 0
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = 0
                                else:
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]  # NA NOVO!!!!
                                    ball.velocity[1] = ball.velocity[0] / ratio
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = ball.velocity[3] / ratio


                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                        else:
                            if abs(ball.velocity[0]) < 0.1:

                                ball.velocity[0] = 0
                                ball.velocity[3] = 0

                                ball.velocity[1] = 0# NA NOVO!!!!
                                ball.velocity[4] = 0

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                                ball.distance_after_rolling[3] = ball.distance[3]
                                ball.time_after_rolling_x = time_array[i]

                            else:
                                y_0 = np.array([ball.distance[0], ball.velocity[0]])
                                solution = solve_ivp(fun_rolling_backwards, (time_array[i], time_array[i + 1]), y_0)
                                ball.distance[0] = solution.y[0][-1]
                                ball.velocity[0] = solution.y[1][-1]
                                ball.velocity[3] = ball.velocity[0] / ball.radius
                                ball.distance[3] = ball.distance_after_sliding[3] + (
                                        ball.distance[0] - ball.distance_after_sliding[0]) / ball.radius

                                if ball.velocity[1] == 0:
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]  # NA NOVO!!!!
                                    ball.velocity[1] = 0
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = 0
                                else:
                                    ball.distance[1] = a / b * ball.velocity[1] + ball.distance[1]  # NA NOVO!!!!
                                    ball.velocity[1] = ball.velocity[0] / ratio
                                    ball.distance[4] = a / b * ball.velocity[4] + ball.distance[4]
                                    ball.velocity[4] = ball.velocity[3] / ratio

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                    elif ball.get_motion_status() == 'not_moving':
                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)
                    else:
                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)
                else:
                    if ball.velocity[0] != 0:
                        ratio = ball.velocity[1] / ball.velocity[0]

                    if i == 0:
                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)
                    if i == len(self.time_array) - 2:
                        print(current_values)
                        print(ball.get_motion_status())
                        print(f'Length of values list: {len(ball.list)}')
                    if ball.get_motion_status() == 'projectile_motion':

                        y_0 = np.array(
                            [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1], ball.distance[2],
                             ball.velocity[2]])
                        solution = solve_ivp(fun_projectile_motion, (time_array[i], time_array[i + 1]), y_0)

                        ball.distance[0] = solution.y[0][-1]
                        ball.velocity[0] = solution.y[1][-1]
                        ball.distance[1] = solution.y[2][-1]
                        ball.velocity[1] = solution.y[3][-1]
                        ball.distance[2] = solution.y[4][-1]
                        ball.velocity[2] = solution.y[5][-1]

                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)


                    elif ball.get_motion_status() == 'collision':
                        if abs(ball.velocity[2] * epsilon) >= 0.1:
                            ball.velocity[2] = ball.velocity[2] * epsilon * (-1)
                            #ball.distance[2] = radius
                            current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                              ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                              ball.distance[4], ball.velocity[4]]
                            ball.list.append(current_values)

                        elif abs(ball.velocity[2] * epsilon) < 0.1:
                            print(f'After collision: {current_values}')
                            # print(f'Time elapsed after last collision is{i * a / b: .2f} s.')
                            # print(f'Distance x after last collision is{ball.get_distance_x(): .2f} m.')
                            ball.velocity[2] = 0
                            ball.distance[2] = ball.radius
                            current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                              ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                              ball.distance[4], ball.velocity[4]]
                            ball.list.append(current_values)
                            ball.time_after_collision = time_array[i]

                        else:
                            print('An Error has occured, regarding collision')
                    elif ball.get_motion_status_y() == 'sliding_y':
                        # if i == 0:
                        # ball.set_time_after_collision(0)
                        if ball.velocity[1] > 0:
                            if abs(ball.velocity[1]) - abs(ball.velocity[4]) * ball.radius < 0.1:
                                ball.velocity[4] = ball.velocity[1] / ball.radius
                                ball.velocity[3] = ball.velocity[0] / ball.radius #NA NOVO!!!!
                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                                """Used to animate ball movement:"""
                                ball.time_after_sliding_x = time_array[i]
                                ball.distance_after_sliding[3] = ball.distance[3]
                                ball.velocity_after_sliding[0] = ball.velocity[0]
                                ball.distance_after_sliding[0] = ball.distance[0]

                                ball.distance_after_sliding[4] = ball.distance[4]#NA NOVO!!!!
                                ball.velocity_after_sliding[1] = ball.velocity[1]
                                ball.distance_after_sliding[1] = ball.distance[1]




                            else:
                                y_0 = np.array([ball.distance[1], ball.velocity[1], ball.distance[4], ball.velocity[4]])
                                solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)
                                ball.distance[1] = solution.y[0][-1]
                                ball.velocity[1] = solution.y[1][-1]
                                ball.distance[4] = solution.y[2][-1]
                                ball.velocity[4] = solution.y[3][-1]

                                ball.distance[0] = a / b * ball.velocity[0] + ball.distance[0]#NA NOVO!!!!
                                ball.velocity[0] = ball.velocity[1] / ratio
                                ball.distance[3] = a / b * ball.velocity[3] + ball.distance[3]
                                ball.velocity[3] = ball.velocity[4] / ratio

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                        else:
                            if abs(ball.velocity[1]) - abs(ball.velocity[4]) * ball.radius < 0.1:
                                ball.velocity[4] = ball.velocity[1] / ball.radius
                                ball.velocity[3] = ball.velocity[0] / ball.radius  # NA NOVO!!!!
                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                                """Used to animate ball movement:"""
                                ball.time_after_sliding = time_array[i]
                                ball.distance_after_sliding[3] = ball.distance[3]
                                ball.velocity_after_sliding[0] = ball.velocity[0]
                                ball.distance_after_sliding[0] = ball.distance[0]

                                ball.distance_after_sliding[4] = ball.distance[4]  # NA NOVO!!!!
                                ball.velocity_after_sliding[1] = ball.velocity[1]
                                ball.distance_after_sliding[1] = ball.distance[1]



                            else:
                                y_0 = np.array([ball.distance[1], ball.velocity[1], ball.distance[4], ball.velocity[4]])
                                solution = solve_ivp(fun_slipping_rolling_backwards, (time_array[i], time_array[i + 1]),
                                                     y_0)
                                ball.distance[1] = solution.y[0][-1]
                                ball.velocity[1] = solution.y[1][-1]
                                ball.distance[4] = solution.y[2][-1]
                                ball.velocity[4] = solution.y[3][-1]

                                ball.distance[0] = a / b * ball.velocity[0] + ball.distance[0]  # NA NOVO!!!!
                                ball.velocity[0] = ball.velocity[1] / ratio
                                ball.distance[3] = a / b * ball.velocity[3] + ball.distance[3]
                                ball.velocity[3] = ball.velocity[4] / ratio

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1],
                                                  ball.velocity[1], ball.distance[2], ball.velocity[2],
                                                  ball.distance[3], ball.velocity[3], ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                    elif ball.get_motion_status_y() == 'rolling_y':

                        if i == 0:
                            # ball.set_time_after_sliding(0)
                            ball.velocity_after_sliding[0] = ball.velocity[0]
                            ball.distance_after_sliding[0] = ball.distance[0]
                            ball.distance_after_sliding[3] = ball.distance[3]

                            ball.velocity_after_sliding[1] = ball.velocity[1]#NA NOVO!!!!!
                            ball.distance_after_sliding[1] = ball.distance[1]
                            ball.distance_after_sliding[4] = ball.distance[4]

                        if ball.velocity[1] > 0:
                            if abs(ball.velocity[1]) < 0.1:

                                ball.velocity[0] = 0
                                ball.velocity[3] = 0

                                ball.velocity[1] = 0 #NA NOVO!!!!
                                ball.velocity[4] = 0

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                                ball.distance_after_rolling[3] = ball.distance[3]
                                ball.distance_after_rolling[4] = ball.distance[4]#NA NOVO!!!
                                ball.time_after_rolling_x = time_array[i]


                            else:
                                y_0 = np.array([ball.distance[1], ball.velocity[1]])
                                solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                                ball.distance[1] = solution.y[0][-1]
                                ball.velocity[1] = solution.y[1][-1]
                                ball.velocity[4] = ball.velocity[1] / ball.radius
                                ball.distance[4] = ball.distance_after_sliding[4] + (
                                        ball.distance[1] - ball.distance_after_sliding[1]) / ball.radius

                                ball.distance[0] = a / b * ball.velocity[0] + ball.distance[0]  # NA NOVO!!!!
                                ball.velocity[0] = ball.velocity[1] / ratio
                                ball.distance[3] = a / b * ball.velocity[3] + ball.distance[3]
                                ball.velocity[3] = ball.velocity[4] / ratio

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                        else:
                            if abs(ball.velocity[1]) < 0.1:

                                ball.velocity[0] = 0
                                ball.velocity[3] = 0

                                ball.velocity[1] = 0# NA NOVO!!!!
                                ball.velocity[4] = 0

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)

                                ball.distance_after_rolling[3] = ball.distance[3]
                                ball.time_after_rolling_x = time_array[i]

                            else:
                                y_0 = np.array([ball.distance[1], ball.velocity[1]])
                                solution = solve_ivp(fun_rolling_backwards, (time_array[i], time_array[i + 1]), y_0)
                                ball.distance[1] = solution.y[0][-1]
                                ball.velocity[1] = solution.y[1][-1]
                                ball.velocity[4] = ball.velocity[1] / ball.radius
                                ball.distance[4] = ball.distance_after_sliding[4] + (
                                        ball.distance[1] - ball.distance_after_sliding[1]) / ball.radius

                                ball.distance[0] = a / b * ball.velocity[0] + ball.distance[0]  # NA NOVO!!!!
                                ball.velocity[0] = ball.velocity[1] / ratio
                                ball.distance[3] = a / b * ball.velocity[3] + ball.distance[3]
                                ball.velocity[3] = ball.velocity[4] / ratio

                                current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                                  ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                                  ball.distance[4], ball.velocity[4]]
                                ball.list.append(current_values)
                    elif ball.get_motion_status() == 'not_moving':
                        # print(f'i = {i}')
                        # print(f'dolžina lista: {len(ball.list)}')
                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)
                    else:
                        current_values = [ball.distance[0], ball.velocity[0], ball.distance[1], ball.velocity[1],
                                          ball.distance[2], ball.velocity[2], ball.distance[3], ball.velocity[3],
                                          ball.distance[4], ball.velocity[4]]
                        ball.list.append(current_values)

            seznam1, collision = self.check_collision()  # list,boolean

            if collision:
                for i in seznam1: #list containing balls in collision
                    first = i[0] #balls in collision
                    second = i[1]
                    for j in self.list:
                        a1 = j.ID
                        if a1 == first.ID:
                            st_ball = j
                        elif a1 == second.ID:
                            nd_ball = j

                    m1 = st_ball.mass
                    v1x = st_ball.velocity[0]
                    v1y = st_ball.velocity[1]
                    v1z = st_ball.velocity[2]

                    m2 = nd_ball.mass
                    v2x = nd_ball.velocity[0]
                    v2y = nd_ball.velocity[1]
                    v2z = nd_ball.velocity[2]
                    """print(f'Pred odbojem:')
                    print(f'v1x = {v1x}')
                    print(f'v1y = {v1y}')
                    print(f'v1z = {v1z}')
                    print(f'v2x = {v2x}')
                    print(f'v2y = {v2y}')
                    print(f'v2z = {v2z}')"""

                    dx = nd_ball.distance[0] - st_ball.distance[0]  # distance between balls
                    dz = nd_ball.distance[2] - st_ball.distance[2]
                    angle = np.arctan2(dz, dx)  # angle between horizontal line and line that goes through both centers
                    """Relative distance to centre of 1st ball:"""
                    x1 = 0
                    z1 = 0
                    x2 = dx * np.cos(angle) + dz * np.sin(angle)
                    z2 = dz * np.cos(angle) - dx * np.sin(angle)
                    """rotating velocity:"""
                    vx1 = v1x * np.cos(angle) + v1z * np.sin(angle)
                    vz1 = v1z * np.cos(angle) - v1x * np.sin(angle)
                    vx2 = v2x * np.cos(angle) + v2z * np.sin(angle)
                    vz2 = v2z * np.cos(angle) - v2x * np.sin(angle)
                    """print(f'Rotirane hitrosti:')
                    print(f'vx1 = {vx1}')
                    print(f'vz1 = {vz1}')
                    print(f'vx2 = {vx2}')
                    print(f'vz2 = {vz2}')"""
                    """resolve 1-D velocity and use temporary variables:"""
                    # vx1final = v_x1k
                    vx1final = vx1 - (vx1 - vx2) * (1 + epsilon_ball) * m2 / (m1 + m2)
                    # vx2final = v_x2k
                    vx2final = vx2 - (vx2 - vx1) * (1 + epsilon_ball) * m1 / (m1 + m2)
                    """update velocity:"""
                    vx1 = vx1final
                    vx2 = vx2final
                    """Fixing overlap:"""
                    absV = abs(vx1) + abs(vx2)
                    overlap = (st_ball.radius + nd_ball.radius) - abs(x1 - x2)
                    x1 += vx1 / absV * overlap
                    x2 += vx2 / absV * overlap
                    """Rotate relative positions back:"""
                    x1final = x1 * np.cos(angle) - z1 * np.sin(angle)
                    z1final = z1 * np.cos(angle) + x1 * np.sin(angle)
                    x2final = x2 * np.cos(angle) - z2 * np.sin(angle)
                    z2final = z2 * np.cos(angle) + x2 * np.sin(angle)
                    """Calculate new absolute positions:"""
                    nd_ball.distance[0] = st_ball.distance[0] + x2final
                    nd_ball.distance[2] = st_ball.distance[2] + z2final

                    st_ball.distance[0] = st_ball.distance[0] + x1final
                    st_ball.distance[2] = st_ball.distance[2] + z1final
                    """rotate velocities back:"""
                    v1xr = vx1 * np.cos(angle) - vz1 * np.sin(angle)
                    v1zr = vz1 * np.cos(angle) + vx1 * np.sin(angle)
                    v2xr = vx2 * np.cos(angle) - vz2 * np.sin(angle)
                    v2zr = vz2 * np.cos(angle) + vx2 * np.sin(angle)

                    """Calculating y velocities:"""
                    v1yr = v1y - (v1y - v2y) * (1 + epsilon_ball) * m2 / (m1 + m2)
                    v2yr = v2y - (v2y - v1y) * (1 + epsilon_ball) * m1 / (m1 + m2)

                    """print(f'Kot je {angle / np.pi * 180}°')
                    print(f'Po odboju:')
                    print(f'v1xr = {v1xr}')
                    print(f'v1yr = {v1yr}')
                    print(f'v1zr = {v1zr}')
                    print(f'v2xr = {v2xr}')
                    print(f'v2yr = {v2yr}')
                    print(f'v2zr = {v2zr}')"""


                    """Setting new values:"""
                    st_ball.velocity[0] = v1xr
                    st_ball.velocity[1] = v1yr
                    st_ball.velocity[2] = v1zr
                    nd_ball.velocity[0] = v2xr
                    nd_ball.velocity[1] = v2yr
                    nd_ball.velocity[2] = v2zr
                    # do tu----------------------------

    def new_values(self):
        for ball in self.list:
            ball.distance[0] = ball.list[-1][0]
            ball.distance[1] = ball.list[-1][2]
            ball.distance[2] = ball.list[-1][4]
            ball.distance[3] = ball.list[-1][6]
            ball.distance[4] = ball.list[-1][8]
            ball.velocity[0] = ball.list[-1][1]
            ball.velocity[1] = ball.list[-1][3]
            ball.velocity[2] = ball.list[-1][5]
            ball.velocity[3] = ball.list[-1][7]
            ball.velocity[4] = ball.list[-1][9]
            ball.list = []
            ball.time_after_collision_x = 0
            ball.time_after_sliding_x = 0
            ball.time_after_rolling_x = 0
            ball.distance_after_sliding = [0, 0, 0, 0, 0]
            ball.distance_after_rolling = [0, 0, 0, 0, 0]
            ball.velocity_after_sliding = [0, 0, 0, 0, 0]
            ball.velocity_after_rolling = [0, 0, 0, 0, 0]
    def draw(self):
        for ball in self.list:
            ID = ball.ID
            plt.plot(time_array, [item[1] for item in ball.list], label='v_x-t')
            plt.xlabel('t[s]');
            plt.ylabel('vx[m/s]');
            plt.title(f'Krogla {ID}');
            plt.legend();
            plt.grid()
            plt.show()
            plt.plot(time_array, [item[7] for item in ball.list], label='omega_x-t')
            plt.xlabel('t[s]');
            plt.ylabel('omega_x [rad/s]');
            plt.title(f'Krogla {ID}');
            plt.legend();
            plt.grid()
            plt.show()
            plt.plot([item[0] for item in ball.list], [item[4] for item in ball.list], label='Pot krogle')
            plt.xlabel('x [m]');
            plt.ylabel('z [m]');
            plt.title(f'Krogla {ID}');
            plt.legend();
            plt.grid()
            plt.show()
            plt.plot(time_array, [item[5] for item in ball.list], label='v_z-t')
            plt.xlabel('t[s]');
            plt.ylabel('vz[m/s]');
            plt.title(f'Krogla {ID}');
            plt.legend();
            plt.grid()
            plt.show()
            plt.plot(time_array, [item[4] for item in ball.list], label='z-t')
            plt.legend();
            plt.xlabel('t[s]');
            plt.ylabel('z[m]');
            plt.title(f'Krogla {ID}');
            plt.grid()
            plt.show()
            plt.plot(time_array, [item[6] for item in ball.list], label='phi_x-t')
            plt.legend();
            plt.xlabel('t[s]');
            plt.ylabel('phi X[rad]');
            plt.title(f'Krogla {ID}');
            plt.grid()
            plt.show()

            """plt.plot(time_array,[item[4] for item in ball.list], label='Velocity_phi')
            plt.show()"""
    def draw_y(self):
        for ball in self.list:
            ID = ball.ID
            plt.plot(time_array, [item[3] for item in ball.list], label='v_y-t')
            plt.xlabel('t[s]');
            plt.ylabel('Vy[m/s]');
            plt.title(f'Krogla {ID}');
            plt.legend();
            plt.grid()
            plt.show()
            plt.plot(time_array, [item[9] for item in ball.list], label='omega_y-t')
            plt.xlabel('t[s]');
            plt.ylabel('omega_y [rad/s]');
            plt.title(f'Krogla {ID}');
            plt.legend();
            plt.grid()
            plt.show()
            plt.plot([item[2] for item in ball.list], [item[4] for item in ball.list], label='Pot krogle')
            plt.xlabel('y [m]');
            plt.ylabel('z [m]');
            plt.title(f'Krogla {ID}');
            plt.legend();
            plt.grid()
            plt.show()
            plt.plot(time_array, [item[8] for item in ball.list], label='phi_y-t')
            plt.legend();
            plt.xlabel('t[s]');
            plt.ylabel('phi_y [rad]');
            plt.title(f'Krogla {ID}');
            plt.grid()
            plt.show()
    def draw_3D(self):
        for ball in self.list:
            ID = ball.ID
            ax = plt.axes(projection='3d')
            ax.set_xlabel('X [m]');
            ax.set_ylabel('Y [m]');
            ax.set_zlabel('Z [m]');
            plt.title(f'Gibanje krogle {ID} v 3D');

            # Data for a three-dimensional line
            zline = [item[4] for item in ball.list]
            xline = [item[0] for item in ball.list]
            yline = [item[2] for item in ball.list]
            ax.plot3D(xline, yline, zline, 'gray')
            plt.show()

    def animatey(self):
        # ANIMATION
        fig = plt.figure()
        ax = plt.axes(xlim=(-50, 10), ylim=(0, 20))  # z lim???
        lines = []
        for i in range(len(self.list)):
            line, = ax.plot([], [], 'o', lw=2, color='b')
            lines.append(line)

        title = ax.text(0.5, 1.05, "Animation of ball movement, y-z axis", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="center")

        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(frame):
            for i in range(len(lines)):
                y_out = [[item[2] for item in self.list[i].list][frame],
                         [item[2] for item in self.list[i].list][frame] + np.sin(
                             [item[8] for item in self.list[i].list][frame]) * radius]
                z_out = [[item[4] for item in self.list[i].list][frame],
                         [item[4] for item in self.list[i].list][frame] + np.cos(
                             [item[6] for item in self.list[i].list][frame]) * radius]
                lines[i].set_data((y_out, z_out))

            time_text.set_text(time_template % (frame * a / b))
            return lines

        anim = FuncAnimation(fig, animate, frames=b, interval=a / b * 1000)
        plt.grid()
        plt.show()
        """for i in self.list:
            print(i.coll)
            print(i.num_colls)"""
    def animate(self):
        # ANIMATION
        fig = plt.figure()
        ax = plt.axes(xlim=(-1.5, 50), ylim=(0, 20))  # z lim???
        lines = []
        for i in range(len(self.list)):
            line, = ax.plot([], [], 'o', lw=2, color='b')
            lines.append(line)

        title = ax.text(0.5, 1.05, "Animation of ball movement, x-z axis", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="center")

        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(frame):
            for i in range(len(lines)):
                x_out = [[item[0] for item in self.list[i].list][frame],
                         [item[0] for item in self.list[i].list][frame] + np.sin(
                             [item[6] for item in self.list[i].list][frame]) * radius]
                z_out = [[item[4] for item in self.list[i].list][frame],
                         [item[4] for item in self.list[i].list][frame] + np.cos(
                             [item[6] for item in self.list[i].list][frame]) * radius]
                lines[i].set_data((x_out, z_out))

            time_text.set_text(time_template % (frame * a / b))
            return lines

        anim = FuncAnimation(fig, animate, frames=b, interval=a / b * 1000)
        plt.grid()
        plt.show()
    def animate_xy(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(-1.5, 20), ylim=(-3, 3))
        lines = []
        for i in range(len(self.list)):
            line, = ax.plot([], [], 'o', lw=2, color='b')
            lines.append(line)

        title = ax.text(0.5, 1.05, "Animation of ball movement, x-y axis",
                        bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="center")

        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(frame):

            for i in range(len(lines)):
                x_out = [[item[0] for item in self.list[i].list][frame]]
                y_out = [[item[2] for item in self.list[i].list][frame]]
                lines[i].set_data((x_out, y_out))

            time_text.set_text(time_template % (frame * a / b))
            return lines

        anim = FuncAnimation(fig, animate, frames=b, interval=a / b * 1000)
        plt.plot([0,15], [2,2])
        plt.plot([0, 15], [-2, -2])
        plt.plot([15, 15], [2, -2])
        plt.grid()
        plt.show()

class Ball:
    def __init__(self, ID, distance, velocity, radius, mass, list):
        self.ID = ID
        self.distance = distance
        self.velocity =velocity
        self.radius = radius
        self.mass = mass
        self.list = list
        self.coll = 1  # gre lahko v kolizijo
        self.num_colls = 0 # število trkov posamezne krogle
        self.time_after_collision_x = 0
        self.time_after_sliding_x = 0
        self.time_after_rolling_x = 0
        self.distance_after_sliding = [0, 0, 0, 0, 0]
        self.distance_after_rolling = [0, 0, 0, 0, 0]
        self.velocity_after_sliding = [0, 0, 0, 0, 0]
        self.velocity_after_rolling = [0, 0, 0, 0, 0]

    def print(self):
        s = "ID=" + str(self.ID) + "DIST X=" + str(self.distance[0]) + " VELOCITY X=" + str(
            self.velocity[0]) + " DIST Y=" + str(self.distance[1]) + " VELOCITY Y=" + str(
            self.velocity[1]) + " DIST Z=" + str(self.distance[2]) + " VELOCITY Z=" + str(
            self.velocity[2]) + " DIST PHI_X=" + str(self.distance[3]) + " VELOCITY PHI_X=" + str(
            self.velocity[3]) + " DIST PHI_Y=" + str(self.distance[4]) + " VELOCITY PHI_Y=" + str(self.velocity[4])
        return s

    def print_list(self):
        print(self.list)

    def get_motion_status(self):
        """
        Function used to determine motion state of the ball.
        """
        state = ""
        if self.velocity[0] == 0 and self.velocity[1] and self.velocity[2] == 0 and self.distance[2] == self.radius:
            state = "not_moving"
        elif self.distance[2] > self.radius or self.velocity[2] > 0:
            state = "projectile_motion"
        elif self.distance[2] <= self.radius and abs(self.velocity[2]) > 0:
            state = "collision"
        elif self.velocity[2] == 0 and self.distance[2] == self.radius and abs(self.velocity[0]) - abs(
                self.velocity[3]) * self.radius > 0.001:
            state = "sliding_x"
        elif self.velocity[2] == 0 and self.distance[2] == self.radius and abs(self.velocity[0]) - abs(
                self.velocity[3]) * self.radius < 0.001 and abs(self.velocity[0]) > 0:
            state = "rolling_x"
        else:
            state = "not_moving_x"

        return state

    def get_motion_status_y(self):
        """
        Function used to determine motion state of the ball.
        """
        state = ""
        if self.velocity[0] == 0 and self.velocity[1] == 0 and self.distance[2] == self.radius:
            state = "not_moving"
        elif self.distance[2] > self.radius or self.velocity[2] > 0:
            state = "projectile_motion"
        elif self.distance[2] <= self.radius and abs(self.velocity[2]) > 0:
            state = "collision"
        elif self.velocity[2] == 0 and self.distance[2] == self.radius and abs(self.velocity[1]) - abs(
                self.velocity[4]) * self.radius > 0.001:
            state = "sliding_y"
        elif self.velocity[2] == 0 and self.distance[2] == self.radius and abs(self.velocity[1]) - abs(
                self.velocity[4]) * self.radius < 0.001 and abs(self.velocity[1]) > 0:
            state = "rolling_y"
        else:
            state = "not_moving_y"

        return state

if __name__ == '__main__':
    tic = time.time()
    H = 1.5
    m = 0.5
    v_0 = 10
    v_1 = 1
    g = 9.81
    x0 = 0
    phi0 = 0
    v_phi0 = 0
    radius = 0.1
    epsilon = 0.5
    epsilon_ball = 0.9
    mi = 0.3
    f = 0.2
    a = 10
    b = 1000
    time_array = np.linspace(0, a, b)

    #initial values:
    radius1 = 0.05
    radius2=0.1
    mass1 = 0.3
    mass2 = 0.5
    #distance1 = [x, y, z, phi_x, phi_z]
    distance1 = [0, 0, 1.5, 0, 0]
    #velocity1 = [v_x, v_y, v_z, omega_x, omega_y]
    velocity1 = [5.0, 0.2, 2, 0, 0]

    distance2 = [0, 0, 1.5, 0, 0]
    velocity2 = [5.1, 0.09, 2, 0, 0]

    distance3 = [0, 0, 1.5, 0, 0]
    velocity3 = [5.1, 0.11, 2, 0, 0]

    distance4 = [0, 0.8, 1.5, 0, 0]
    velocity4 = [4.5, 0.09, 1.8, 0, 0]

    distance5 = [0, 0.13, 1.5, 0, 0]
    velocity5 = [5.0, 0.2, 2, 0, 0]

    distance6 = [0, 0, 1.5, 0, 0]
    velocity6 = [8.4, 1, 2.5, 0, 0]

    distance7 = [0, 0, 1.5, 0, 0]
    velocity7 = [6.4, -0.35, 1.5, 0, 0]

    ball1 = Ball(1, distance1, velocity1, radius1, mass1, [])
    ball2 = Ball(2, distance2, velocity2, radius2, mass2, [])
    ball3 = Ball(3, distance3, velocity3, radius2, mass2, [])
    ball4 = Ball(4, distance4, velocity4, radius2, mass2, [])
    ball5 = Ball(5, distance5, velocity5, radius2, mass2, [])
    ball6 = Ball(6, distance6, velocity6, radius2, mass2, [])
    ball7 = Ball(7, distance7, velocity7, radius2, mass2, [])
