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


class Balls:
    def __init__(self,list,time_array):
        self.time_array=time_array
        self.list=[]
        for i in list:
            self.add_ball(i)
    def add_ball(self, ball):
        self.list.append(ball)

    def print(self):
        for i in self.list:
            i.print()

    def time_solution(self):
        values_array = []
        for i in range(len(self.time_array) - 1):
            for ball in self.list:
                if i==0:
                    current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                      ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                    ball.list.append(current_values)
                if ball.get_motion_status() == 'projectile_motion':

                    y_0 = np.array(
                        [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(), ball.get_velocity_y()])
                    solution = solve_ivp(fun_projectile_motion, (time_array[i], time_array[i + 1]), y_0)
                    ball.set_distance_x(solution.y[0][-1])
                    ball.set_velocity_x(solution.y[1][-1])
                    ball.set_distance_y(solution.y[2][-1])
                    ball.set_velocity_y(solution.y[3][-1])

                    current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                      ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                    ball.list.append(current_values)


                elif ball.get_motion_status() == 'collision':
                    if abs(ball.get_velocity_y() * epsilon) >= 0.1:
                        ball.set_velocity_y(ball.get_velocity_y() * epsilon * (-1))
                        ball.set_distance_y(radius)
                        current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                          ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                        ball.list.append(current_values)

                    elif abs(ball.get_velocity_y() * epsilon) < 0.1:
                        print(f'Time elapsed after last collision is{i * a / b: .2f} s.')
                        print(f'Distance x after last collision is{ball.get_distance_x(): .2f} m.')
                        ball.set_velocity_y(0)
                        ball.set_distance_y(radius)
                        current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                          ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                        ball.list.append(current_values)
                        # time_after_collision.append(time_array[i])
                        ball.set_time_after_collision(time_array[i])
                        print(f'!!!!!!!!!!!!!! {ball.get_time_after_collision()}')

                    else:
                        print('An Error has occured, regarding collision')
                elif ball.get_motion_status() == 'sliding':
                    if i == 0:
                        # time_after_collision.append(0) #Če se gibanje začne z drsnim trenjem
                        ball.set_time_after_collision(0)
                    if abs(ball.get_velocity_x()) - abs(ball.get_velocity_phi()) * ball.get_radius() < 0.1:
                        print(f'Time elapsed after sliding is{i * a / b: .2f} s.')
                        print(f'Distance x after sliding is{ball.get_distance_x(): .2f} m.')
                        ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                        current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                          ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                        ball.list.append(current_values)

                        # time_after_sliding.append(time_array[i])
                        # distance_phi_after_sliding.append(ball.get_distance_phi())
                        # velocity_x_after_sliding.append(ball.get_velocity_x())
                        # distance_x_after_sliding.append(ball.get_distance_x())
                        """Used to animate ball movement:"""
                        ball.set_time_after_sliding(time_array[i])
                        ball.set_distance_phi_after_sliding(ball.get_distance_phi())
                        ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                        ball.set_distance_x_after_sliding(ball.get_distance_x())
                        print(f'!!!!!!!!!!!!!! {ball.get_time_after_sliding()}')
                        print(f'!!!!!!!!!!!!!! v_x= {ball.get_velocity_x_after_sliding()}')
                        print(f'!!!!!!!!!!!!!! v_phi= {ball.get_velocity_phi()}')


                    else:
                        y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_phi(),
                                        ball.get_velocity_phi()])
                        solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)
                        ball.set_distance_x(solution.y[0][-1])
                        ball.set_velocity_x(solution.y[1][-1])
                        ball.set_distance_phi(solution.y[2][-1])
                        ball.set_velocity_phi(solution.y[3][-1])

                        # distance_phi = (5 * mi * g) / (4 * ball.get_radius()) * (time_array[i]- time_after_collision)**2
                        distance_phi = (5 * mi * g) / (4 * ball.get_radius()) * (
                                time_array[i] - ball.get_time_after_collision()) ** 2
                        ball.set_distance_phi(distance_phi)

                        current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                          ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                        ball.list.append(current_values)

                elif ball.get_motion_status() == 'rolling':
                    if i == 0:
                        # time_after_sliding.append(0) #če ni drsenja(ni še upoštevano, če je kotaljenje po poševnem metu!)
                        # velocity_x_after_sliding.append(ball.get_velocity_x())
                        # distance_x_after_sliding.append(ball.get_distance_x())

                        ball.set_time_after_sliding(
                            0)  # če ni drsenja(ni še upoštevano, če je kotaljenje po poševnem metu!)
                        ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                        ball.set_distance_x_after_sliding(ball.get_distance_x())

                    if abs(ball.get_velocity_x()) < 0.1:

                        ball.set_velocity_x(0)
                        ball.set_velocity_phi(0)
                        current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                          ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                        ball.list.append(current_values)

                        # distance_phi_after_rolling.append(ball.get_distance_phi())
                        # time_after_rolling.append(time_array[i])

                        ball.set_distance_phi_after_rolling(ball.get_distance_phi())
                        ball.set_time_after_rolling(time_array[i])

                        print(f'!!!!!!!!!!!!!! {ball.get_time_after_rolling()}')

                        print(f'Time elapsed after rolling is{i * a / b: .2f} s.')
                        print(f'Distance x after rolling is{ball.get_distance_x(): .2f} m.')
                        print(f'Distance_phi after rolling: {ball.get_distance_phi()} rad.')
                        print(f'Type distance_x:{type(ball.get_distance_x())}')

                    else:
                        y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x()])
                        solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                        ball.set_distance_x(solution.y[0][-1])
                        ball.set_velocity_x(solution.y[1][-1])
                        ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                        # ball.set_distance_phi(distance_phi_after_sliding[0][0]+(ball.get_distance_x()-distance_x_after_sliding[0])/ball.get_radius())
                        ball.set_distance_phi(ball.get_distance_phi_after_sliding() + (
                                ball.get_distance_x() - ball.get_distance_x_after_sliding()) / ball.get_radius())
                        current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                          ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                        ball.list.append(current_values)
                elif ball.get_motion_status() == 'not_moving':
                    current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                      ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                    ball.list.append(current_values)
                else:
                    print('Error')

        # return values_array  # values_array treba spremenit v 3-D

    def draw(self):
        for ball in self.list:
            plt.plot(time_array, [item[1] for item in ball.list], label='Velocity_x')
            plt.show()
            plt.plot(time_array, [item[3] for item in ball.list], label='Velocity_y')
            plt.show()
            plt.plot(time_array, [item[2] for item in ball.list], label='Distance_y')
            plt.legend();
            plt.xlabel('t[s]');
            plt.ylabel('vx,vy[m/s], y[m]');
            plt.title('Ball movement');
            plt.xlim([0, np.max(b1.get_time_after_rolling()) + 3])
            plt.show()
    def animate(self):
        # ANIMATION
        fig = plt.figure()
        ax = plt.axes(xlim=(-1.5 + 0, 200 + 10),
                      ylim=(0, 100 + 2))
        lines=[]
        for i in range(len(self.list)):
            line, = ax.plot([], [], 'o', lw=2, color='b')
            lines.append(line)

        title = ax.text(0.5, 1.05, "Animation of ball movement", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="center")

        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(frame):
            print("fdssdfsdffsdddddddddddddddddddddddddddddddddddddddd")
            for i in range(len(lines)):
                x_out = [[item[0] for item in self.list[i].list][frame],[item[0] for item in self.list[i].list][frame] + np.sin([item[4] for item in self.list[i].list][frame]) * radius]
                y_out = [[item[2] for item in self.list[i].list][frame],[item[2] for item in self.list[i].list][frame] + np.cos([item[4] for item in self.list[i].list][frame]) * radius]
                lines[i].set_data((x_out, y_out))

            time_text.set_text(time_template % (frame * a / b))
            return lines

        anim = FuncAnimation(fig, animate, frames=10000, interval=a / b * 1000)
        plt.grid()
        plt.show()

class Ball:
    def __init__(self, ID, distance_x, velocity_x, distance_y, velocity_y, distance_phi, velocity_phi, radius, mass,list):
        self.ID = ID
        self.distance_x = distance_x
        self.velocity_x = velocity_x
        self.distance_y = distance_y
        self.velocity_y = velocity_y
        self.distance_phi = distance_phi
        self.velocity_phi = velocity_phi
        self.radius = radius
        self.mass = mass
        self.list=list
    def print(self):
        print(f"dis_x {self.distance_x} vel_x {self.velocity_x}")
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

    def set_distance_x(self, distance_x):
        self.distance_x = distance_x

    def set_velocity_x(self, velocity_x):
        self.velocity_x = velocity_x

    def set_distance_y(self, distance_y):
        self.distance_y = distance_y

    def set_velocity_y(self, velocity_y):
        self.velocity_y = velocity_y

    def set_distance_phi(self, distance_phi):
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
        elif self.velocity_y == 0 and self.distance_y == self.radius and abs(self.velocity_x) - abs(
                self.velocity_phi) * self.radius > 0.001:
            state = "sliding"
        elif self.velocity_y == 0 and self.distance_y == self.radius and abs(self.velocity_x) - abs(
                self.velocity_phi) * self.radius < 0.001 and abs(self.velocity_x) > 0:
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

    def set_distance_phi_after_sliding(self, distance_phi_after_sliding):
        self.distance_phi_after_sliding = distance_phi_after_sliding

    def get_distance_phi_after_sliding(self):
        return self.distance_phi_after_sliding

    def set_distance_x_after_sliding(self, distance_x_after_sliding):
        self.distance_x_after_sliding = distance_x_after_sliding

    def get_distance_x_after_sliding(self):
        return self.distance_x_after_sliding

    def set_distance_phi_after_rolling(self, distance_phi_after_rolling):
        self.distance_phi_after_rolling = distance_phi_after_rolling

    def get_distance_phi_after_rolling(self):
        return self.distance_phi_after_rolling

    def set_velocity_x_after_sliding(self, velocity_x_after_sliding):
        self.velocity_x_after_sliding = velocity_x_after_sliding

    def get_velocity_x_after_sliding(self):
        return self.velocity_x_after_sliding

if __name__ == '__main__':
    H = 1.5
    m = 0.5
    v_0 = 30
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
    a = 20
    b = 10000
    time_array = np.linspace(0, a, b)

    #INITIALIZATION
    b1 = Ball(1, 0, v_0 * np.cos(alpha_0), H, v_0 * np.sin(alpha_0), 0, 0, radius, 1,[])
    b2 = Ball(2, 10,2* v_0 * np.cos(alpha_0), H, 2 * v_0 * np.sin(alpha_0), 0, 0, radius, 1,[])
    b3 = Ball(2, 100, 2 * v_0 * np.cos(alpha_0), H, 2 * v_0 * np.sin(alpha_0), 0, 0, radius, 1, [])


    Zogemaroge = Balls([b1,b2,b3],time_array)#####ADDING BALLS TO NEW CLASS
    Zogemaroge.time_solution()#####FILING TABLE OF ALL BALLS WITH 6 VARIABLES
    Zogemaroge.draw()#####GAPHS DISPLAY
    Zogemaroge.animate()#####ANIMATION DISPLAY