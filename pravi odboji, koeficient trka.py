import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import pandas as pd


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


def fun_slipping_rolling_backwards(t, y):
    """
    Function is used to solve differential equations,
    while the object is slipping and rolling in negative x direction.
    """
    return y[1], mi * g, y[3], (5 * -mi * g) / (2 * radius)


def fun_rolling(t, y):
    """
    Function is used to solve differential equations,
    while the object is rolling.
    """
    return y[1], -5 / 7 * g * theta


def fun_rolling_backwards(t, y):
    """
    Function is used to solve differential equations,
    while the object is rolling in negative x direction.
    """
    return y[1], 5 / 7 * g * theta


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
        bul=False
        seznam=[]
        brek=False
        for i in self.list:
            if brek:
                break
            for j in self.list:
                id1=i.get_ID()
                id2=j.get_ID()
                if id1==id2:
                    continue
                x1=i.get_distance_x()
                y1=i.get_distance_y()
                rad1=i.get_radius()
                x2=j.get_distance_x()
                y2=j.get_distance_y()
                rad2=j.get_radius()
                if i.coll==1 and j.coll==1:
                    if np.sqrt((x1-x2)**2+(y1-y2)**2) < rad1 + rad2:
                        i.coll=0
                        j.coll=0
                        i.num_colls=i.num_colls+1
                        j.num_colls=j.num_colls+1
                        seznam.append([i,j])
                        bul=True
                        brek=True
                        break
                else:
                    i.coll=1
                    j.coll=1
        return seznam,bul





    def time_solution(self,a,b):
        """Function that computes motion parameters for each ball, at each time interval"""
        for i in range(len(self.time_array) - 1):
            for ball in self.list:
                if i == 0:
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
                        #print(f'Time elapsed after last collision is{i * a / b: .2f} s.')
                        #print(f'Distance x after last collision is{ball.get_distance_x(): .2f} m.')
                        ball.set_velocity_y(0)
                        ball.set_distance_y(radius)
                        current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                          ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                        ball.list.append(current_values)
                        ball.set_time_after_collision(time_array[i])

                    else:
                        print('An Error has occured, regarding collision')
                elif ball.get_motion_status() == 'sliding':
                    #if i == 0:
                        #ball.set_time_after_collision(0)
                    if ball.get_velocity_x() > 0:
                        if abs(ball.get_velocity_x()) - abs(ball.get_velocity_phi()) * ball.get_radius() < 0.1:
                            """print(f'Time elapsed after sliding is{i * a / b: .2f} s.')
                            print(f'Distance x after sliding is{ball.get_distance_x(): .2f} m.')
                            print(f'v_x: {ball.get_velocity_x()} m/s.')
                            print(f'v_phi: {ball.get_velocity_phi()} rad/s.')
                            print(f'Distance phi after sliding: {ball.get_distance_phi()} rad.')"""
                            ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                            current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                              ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                            ball.list.append(current_values)

                            """Used to animate ball movement:"""
                            ball.set_time_after_sliding(time_array[i])
                            ball.set_distance_phi_after_sliding(ball.get_distance_phi())
                            ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                            ball.set_distance_x_after_sliding(ball.get_distance_x())



                        else:
                            y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_phi(),
                                            ball.get_velocity_phi()])
                            solution = solve_ivp(fun_slipping_rolling, (time_array[i], time_array[i + 1]), y_0)
                            ball.set_distance_x(solution.y[0][-1])
                            ball.set_velocity_x(solution.y[1][-1])
                            ball.set_distance_phi(solution.y[2][-1])
                            ball.set_velocity_phi(solution.y[3][-1])

                            current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                              ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                            ball.list.append(current_values)
                    else:
                        if abs(ball.get_velocity_x()) - abs(ball.get_velocity_phi()) * ball.get_radius() < 0.1:
                            """print(f'Time elapsed after sliding is{i * a / b: .2f} s.')
                            print(f'Distance x after sliding is{ball.get_distance_x(): .2f} m.')
                            print(f'v_x: {ball.get_velocity_x()} m/s.')
                            print(f'v_phi: {ball.get_velocity_phi()} rad/s.')
                            print(f'Distance phi after sliding: {ball.get_distance_phi()} rad.')"""
                            ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                            current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                              ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                            ball.list.append(current_values)

                            """Used to animate ball movement:"""
                            ball.set_time_after_sliding(time_array[i])
                            ball.set_distance_phi_after_sliding(ball.get_distance_phi())
                            ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                            ball.set_distance_x_after_sliding(ball.get_distance_x())



                        else:
                            y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_phi(),
                                            ball.get_velocity_phi()])
                            solution = solve_ivp(fun_slipping_rolling_backwards, (time_array[i], time_array[i + 1]),
                                                 y_0)
                            ball.set_distance_x(solution.y[0][-1])
                            ball.set_velocity_x(solution.y[1][-1])
                            ball.set_distance_phi(solution.y[2][-1])
                            ball.set_velocity_phi(solution.y[3][-1])

                            current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                              ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                            ball.list.append(current_values)


                elif ball.get_motion_status() == 'rolling':
                    if i == 0:
                        #ball.set_time_after_sliding(0)
                        ball.set_velocity_x_after_sliding(ball.get_velocity_x())
                        ball.set_distance_x_after_sliding(ball.get_distance_x())
                        ball.set_distance_phi_after_sliding(ball.get_distance_phi())

                    if ball.get_velocity_x() > 0:
                        if abs(ball.get_velocity_x()) < 0.1:

                            ball.set_velocity_x(0)
                            ball.set_velocity_phi(0)
                            current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                              ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                            ball.list.append(current_values)
                            ball.set_distance_phi_after_rolling(ball.get_distance_phi())
                            ball.set_time_after_rolling(time_array[i])

                            """print(f'Time elapsed after rolling is{i * a / b: .2f} s.')
                            print(f'Distance x after rolling is{ball.get_distance_x(): .2f} m.')
                            print(f'Distance_phi after rolling: {ball.get_distance_phi()} rad.')
                            print(f'----------------------------------------------------------------')"""


                        else:
                            y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x()])
                            solution = solve_ivp(fun_rolling, (time_array[i], time_array[i + 1]), y_0)
                            ball.set_distance_x(solution.y[0][-1])
                            ball.set_velocity_x(solution.y[1][-1])
                            ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
                            ball.set_distance_phi(ball.get_distance_phi_after_sliding() + (
                                        ball.get_distance_x() - ball.get_distance_x_after_sliding()) / ball.get_radius())
                            current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                              ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                            ball.list.append(current_values)
                    else:
                        if abs(ball.get_velocity_x()) < 0.1:

                            ball.set_velocity_x(0)
                            ball.set_velocity_phi(0)
                            current_values = [ball.get_distance_x(), ball.get_velocity_x(), ball.get_distance_y(),
                                              ball.get_velocity_y(), ball.get_distance_phi(), ball.get_velocity_phi()]
                            ball.list.append(current_values)

                            ball.set_distance_phi_after_rolling(ball.get_distance_phi())
                            ball.set_time_after_rolling(time_array[i])

                            """print(f'Time elapsed after rolling is{i * a / b: .2f} s.')
                            print(f'Distance x after rolling is{ball.get_distance_x(): .2f} m.')
                            print(f'Distance_phi after rolling: {ball.get_distance_phi()} rad.')
                            print(f'----------------------------------------------------------------')"""


                        else:
                            y_0 = np.array([ball.get_distance_x(), ball.get_velocity_x()])
                            solution = solve_ivp(fun_rolling_backwards, (time_array[i], time_array[i + 1]), y_0)
                            ball.set_distance_x(solution.y[0][-1])
                            ball.set_velocity_x(solution.y[1][-1])
                            ball.set_velocity_phi(ball.get_velocity_x() / ball.get_radius())
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
                    print(ball.print()+" "+str(ball.num_colls))

            seznam1,coll=self.check_collision() #list,boolean(vsaj ena kolizija)

            if coll:
                print(seznam1[0][0].print())
                print(seznam1[0][1].print())
                print(len(seznam1))

                #print(seznam1[0][0].print())

                for i in seznam1:
                    prva=i[0]
                    druga=i[1]
                    #tocki v koliziji
                    st_ball=0
                    nd_ball=0
                    #
                    for j in self.list:
                        a1=j.get_ID()
                        if a1==prva.get_ID():
                            st_ball=j
                        elif a1==druga.get_ID():
                            nd_ball=j

                    m1=st_ball.get_mass()
                    v1x = st_ball.get_velocity_x()
                    v1y = st_ball.get_velocity_y()

                    m2 = nd_ball.get_mass()
                    v2x = nd_ball.get_velocity_x()
                    v2y = nd_ball.get_velocity_y()
                    print(f'Pred odbojem:')
                    print(f'v1x = {v1x}')
                    print(f'v1y = {v1y}')
                    print(f'v2x = {v2x}')
                    print(f'v2y = {v2y}')

                    dx = nd_ball.get_distance_x() - st_ball.get_distance_x() #distance between balls
                    dy = nd_ball.get_distance_y() - st_ball.get_distance_y()
                    angle = np.arctan2(dy,dx) #angle between vertical line and line that goes through both centers
                    """Relative distance to centre of 1st ball:"""
                    x1 = 0
                    y1 = 0
                    x2 = dx * np.cos(angle) + dy * np.sin(angle)
                    y2 = dy * np.cos(angle) - dx * np.sin(angle)
                    """rotating velocity:"""
                    vx1 = v1x * np.cos(angle) + v1y * np.sin(angle)
                    vy1 = v1y * np.cos(angle) - v1x * np.sin(angle)
                    vx2 = v2x * np.cos(angle) + v2y * np.sin(angle)
                    vy2 = v2y * np.cos(angle) - v2x * np.sin(angle)
                    print(f'Rotirane hitrosti:')
                    print(f'vx1 = {vx1}')
                    print(f'vy1 = {vy1}')
                    print(f'vx2 = {vx2}')
                    print(f'vy2 = {vy2}')
                    """resolve 1-D velocity and use temporary variables:"""
                    #vx1final = ((m1 - m2) * vx1 + 2 * m2*vx2) / (m1 + m2)
                    vx1final = vx1 - (vx1 - vx2) * (1 + epsilon_ball) * m2 / (m1 + m2)
                    #vx2final = ((m2 - m1) * vx2 + 2 * m1*vx1) / (m1 + m2)
                    vx2final = vx2 - (vx2 - vx1) * (1 + epsilon_ball) * m1 / (m1 + m2)
                    print(f'Izračunane vx final:')
                    print(f'vx1final = {vx1final}')
                    print(f'vx2final = {vx2final}')


                    """update velocity:"""
                    vx1 = vx1final
                    vx2 = vx2final
                    """Fixing overlap:"""
                    absV = abs(vx1) + abs(vx2)
                    overlap = (st_ball.get_radius() + nd_ball.get_radius()) - abs(x1 - x2)
                    x1 += vx1 / absV * overlap
                    x2 += vx2 / absV * overlap
                    """Rotate relative positions back:"""
                    x1final = x1 * np.cos(angle) - y1 * np.sin(angle)
                    y1final = y1 * np.cos(angle) + x1 * np.sin(angle)
                    x2final = x2 * np.cos(angle) - y2 * np.sin(angle)
                    y2final = y2 * np.cos(angle) + x2 * np.sin(angle)
                    """Calculate new absolute positions:"""
                    nd_ball.set_distance_x(st_ball.get_distance_x() + x2final)
                    nd_ball.set_distance_y(st_ball.get_distance_y() + y2final)

                    st_ball.set_distance_x(st_ball.get_distance_x() + x1final)
                    st_ball.set_distance_y(st_ball.get_distance_y() + y1final)
                    """rotate velocities back:"""
                    v1xr = vx1 * np.cos(angle) - vy1 * np.sin(angle)
                    v1yr = vy1 * np.cos(angle) + vx1 * np.sin(angle)
                    v2xr = vx2 * np.cos(angle) - vy2 * np.sin(angle)
                    v2yr = vy2 * np.cos(angle) + vx2 * np.sin(angle)
                    print(f'Kot je {angle/np.pi*180}°')
                    print(f'Po odboju:')
                    print(f'v1xr = {v1xr}')
                    print(f'v1yr = {v1yr}')
                    print(f'v2xr = {v2xr}')
                    print(f'v2yr = {v2yr}')


                    """st_ball.coll_x=x1
                    st_ball.coll_y=y1
                    nd_ball.coll_x=x2
                    nd_ball.coll_y=y2"""
                    st_ball.set_velocity_x(v1xr)
                    st_ball.set_velocity_y(v1yr)
                    nd_ball.set_velocity_x(v2xr)
                    nd_ball.set_velocity_y(v2yr)
                    """
                    if v1yr<v2yr:
                        if abs(v1yr)<0.1:
                            st_ball.set_velocity_y(0)
                    else:
                        if abs(v2yr) < 0.1:
                            nd_ball.set_velocity_y(0)
                    """


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
            plt.xlim([0, np.max(b1.get_time_after_rolling())])

            """plt.plot(time_array,[item[4] for item in ball.list], label='Velocity_phi')
            plt.show()"""

    def animate(self):
        # ANIMATION
        fig = plt.figure()
        ax = plt.axes(xlim=(-1.5 , 50),
                      ylim=(0, 20 ))
        lines = []
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
            for i in range(len(lines)):
                x_out = [[item[0] for item in self.list[i].list][frame],
                         [item[0] for item in self.list[i].list][frame] + np.sin(
                             [item[4] for item in self.list[i].list][frame]) * radius]
                y_out = [[item[2] for item in self.list[i].list][frame],
                         [item[2] for item in self.list[i].list][frame] + np.cos(
                             [item[4] for item in self.list[i].list][frame]) * radius]
                lines[i].set_data((x_out, y_out))

            time_text.set_text(time_template % (frame * a / b))
            return lines

        anim = FuncAnimation(fig, animate, frames=b, interval=a / b * 1000)
        plt.grid()
        plt.show()
        """for i in self.list:
            print(i.coll)
            print(i.num_colls)"""


class Ball:
    def __init__(self, ID, distance_x, velocity_x, distance_y, velocity_y, distance_phi, velocity_phi, radius, mass,
                 list):
        self.ID = ID
        self.distance_x = distance_x
        self.velocity_x = velocity_x
        self.distance_y = distance_y
        self.velocity_y = velocity_y
        self.distance_phi = distance_phi
        self.velocity_phi = velocity_phi
        self.radius = radius
        self.mass = mass
        self.list = list
        self.coll_x=0
        self.coll_y=0
        self.coll=1 #gre lahko v kolizijo
        self.num_colls=0
        self.time_after_collision = 0
        self.time_after_sliding = 0
        self.distance_phi_after_sliding = 0
        self.distance_x_after_sliding = 0

    def print(self):
        s="ID="+str(self.ID)+"DIST X="+str(self.distance_x)+" VELOCITY X="+str(self.velocity_x)+" DIST Y="+str(self.distance_y)+" VELOCITY Y="+str(self.velocity_y)+" DIST PHI="+str(self.distance_phi)+" VELOCITY PHI="+str(self.velocity_phi)
        return s
    def get_ID(self):
        return self.ID
    def get_list(self):
        return self.list
    def print_list(self):
        print(self.list)

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
        state=""
        if self.velocity_x == 0 and self.distance_y == self.radius:
            state = "not_moving"
        elif self.distance_y > self.radius or self.velocity_y > 0:
            state = "projectile_motion"
        elif self.distance_y <= self.radius and abs(self.velocity_y) > 0:
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
    tic = time.time()
    H = 1.5
    m = 0.5
    v_0 = 10
    g = 9.81
    alpha_0 = 45 * np.pi / 180
    alpha_1 = 45 * np.pi / 180
    alpha_2 = 125 * np.pi / 180
    alpha_4 = 135 * np.pi / 180
    alpha_5 = 90 * np.pi / 180
    x0 = 0
    phi0 = 0
    v_phi0 = 0
    radius = 0.1
    epsilon = 0.5
    epsilon_ball = 1
    mi = 0.3
    f = 0.2
    theta = f / radius
    a = 10
    b = 1000
    time_array = np.linspace(0, a, b)

    # INITIALIZATION
    b1 = Ball(1, 0, v_0 * np.cos(alpha_0), H*4, v_0 * np.sin(alpha_0), 0, 0, radius, 1, [])
    b2 = Ball(2, 2, v_0 * np.cos(alpha_1), H, v_0 * np.sin(alpha_1), 0, 0, radius, 1, [])
    b3 = Ball(3, 25, v_0 * np.cos(alpha_2), H*2, v_0 * np.sin(alpha_1), 0, 0, radius, 1, [])
    b4 = Ball(4, 50, v_0 * np.cos(alpha_4), H , v_0 * np.sin(alpha_4), 0, 0, radius, 1, [])
    b5 = Ball(5, 0, v_0 * np.cos(alpha_5), H , v_0 * np.sin(alpha_5), 0, 0, radius, 1, [])
    b6 = Ball(6, 0, v_0 * np.cos(alpha_5), 2*H, v_0 * np.sin(alpha_5), 0, 0, radius, 1, [])
    b7 = Ball(7 ,30, 0, radius,0, 0, 0, radius, 1, [])
    b8 = Ball(8, 0, v_0 * np.cos(alpha_0), H , v_0 * np.sin(alpha_0), 0, 0, radius, 1, [])
    b9 = Ball(9, 15, v_0 * np.cos(alpha_4), H-0.13 , v_0 * np.sin(alpha_4), 0, 0, radius, 1, [])


    #Zogemaroge = Balls([b1,b2,b3], time_array)  #####ADDING BALLS TO NEW CLASS
    Zogemaroge = Balls([b8, b9], time_array)
    #Zogemaroge = Balls([b1, b7], time_array)  #####ADDING BALLS TO NEW CLASS ena žoga je pri meru
    #Zogemaroge = Balls([b1], time_array)  #####ADDING BALLS TO NEW CLASS
    #Zogemaroge = Balls([], time_array)  #####ADDING BALLS TO NEW CLASS
    #Zogemaroge = Balls([b5, b6], time_array)  #####ADDING BALLS TO NEW CLASS NAVPIČNO
    Zogemaroge.time_solution(a,b)  #####FILING TABLE OF ALL BALLS WITH 6 VARIABLES
    tok = time.time()
    computing_time = tok - tic
    print(f'Computing time = {computing_time: .2f}s.')

    Zogemaroge.animate()  #####ANIMATION DISPLAY




