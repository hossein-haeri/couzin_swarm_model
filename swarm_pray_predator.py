import numpy as np
from numpy.linalg import *
from math import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
import time
import pylab

mpl.use('TkAgg')
visualization = 'on'   # on/off
dimension = '3d'        # 2d/3d
n = 30                # number of agents
max_steps = 2000          # maximum simulation steps
dt = 0.5                # simulation time step size
free_offset = 10        # free space margin (just for visualization)
number_of_alives = n    # initial number of alive individuals
number_of_sharks = 2    # number of sharks
number_of_foods = 3

# set the social behavior vs. threat-escaping behavior ratio
#alpha = .5          # alpha = 1 means it only has social behavior and
                    # alpha = 0 means it only has escape behavior

# alpha_list = np.linspace(0.3, 1, num=50)
alpha_list = [0.5]
number_of_alives_list = []

# Couzin's repulsion/orientation/attraction radii (agent-agent interaction parameters)
r_r = 2
r_o = 10
r_a = 40

# agent-environment parameters
r_thr = 30      # zone of threat (individuals see threats closer than r_thr)
r_res = 100      # zone of resource (individuals see resources closer than r_thr)
r_lethal = 1    # individuals die if they are closer than r_lethal to any threat

field_of_view = 3*pi/2      # individuals'field of vision
field_of_view_shark = 2*pi  # sharks'field of vision

theta_dot_max = 0.5           # maximum angular velocity of the individuals
theta_dot_max_shark = .3    # maximum angular velocity of the sharks

constant_speed = 2          # translational velocity of the individuals
shark_speed = 5             # translational velocity of the sharks
food_speed = 0              # translational velocity of the resources

area_width = 50   # x_max[m]
area_height = 50  # y_max[m]
area_depth = 50   # z_max[m]

# np.seterr(divide='ignore', invalid='ignore')

class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        self.pos = np.array([0, 0, 0])
        self.pos[0] = np.random.uniform(0, area_width)
        self.pos[1] = np.random.uniform(0, area_height)
        self.pos[2] = np.random.uniform(0, area_depth)
        self.vel = np.random.uniform(-1, 1, 3)
        self.is_alive = 1
        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0
        self.vel = self.vel / norm(self.vel) * speed

    def update_position(self, delta_t):
        if self.is_alive == 1:
            self.pos = self.pos + self.vel * delta_t



def rotation_matrix_about(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


if __name__ == '__main__':


    start = time.time()


    for alpha in alpha_list:
        # initialize the variables
        swarm = []
        sharks = []
        foods = []
        swarm_pos = np.zeros([n, 3])
        swarm_vel = np.zeros([n, 3])
        swarm_color = np.zeros(n)
        sharks_pos = np.zeros([number_of_sharks, 3])
        sharks_vel = np.zeros([number_of_sharks, 3])
        d = np.array([0, 0, 0])
        d_social = np.array([0, 0, 0])
        d_r = np.array([0, 0, 0])
        d_o = np.array([0, 0, 0])
        d_a = np.array([0, 0, 0])
        d_thr = np.array([0, 0, 0])
        d_res = np.array([0, 0, 0])
        t = 0

        # create 'swarm' and 'shark' and 'food' instances based on the Agent class
        [swarm.append(Agent(i, constant_speed)) for i in range(n)]
        [sharks.append(Agent(i, shark_speed)) for i in range(number_of_sharks)]
        [foods.append(Agent(i, food_speed)) for i in range(number_of_foods)]

        # initialize the figure
        if visualization is 'on':
            fig = plt.figure()
            if dimension == '3d':
                ax = fig.gca(projection='3d')
            else:
                ax = fig.gca()

        # begin the simulation loop
        number_of_alives = n

        while t < max_steps:

            for i in range(len(swarm)):
                swarm_pos[i, :] = swarm[i].pos
                swarm_vel[i, :] = swarm[i].vel
                swarm_color[i] = 2 - swarm[i].is_alive
            for i in range(number_of_sharks):
                sharks_pos[i, :] = sharks[i].pos
                sharks_vel[i, :] = sharks[i].vel/norm(sharks[i].vel) / 80 * area_width

            t = t + 1

            if visualization is 'on':
                ax.clear()
                if dimension == '2d':
                    # ax.quiver(swarm_pos[:, 0], swarm_pos[:, 1],
                    #           swarm_vel[:, 0], swarm_vel[:, 1],
                    #           swarm_color)

                    pylab.quiver(swarm_pos[:, 0], swarm_pos[:, 1],
                              swarm_vel[:, 0], swarm_vel[:, 1],
                              swarm_color)

                    ax.plot(sharks_pos[:, 0], sharks_pos[:, 1], 'o', color='#FF0000')

                    for each_food in foods:
                        ax.scatter(each_food.pos[0], each_food.pos[1], color='#228B22')

                    ax.set_aspect('equal', 'box')
                    ax.set_xlim(0, area_width)
                    ax.set_ylim(0, area_height)

                else:
                    q = ax.quiver(swarm_pos[:, 0], swarm_pos[:, 1], swarm_pos[:, 2],
                                  swarm_vel[:, 0], swarm_vel[:, 1], swarm_vel[:, 2])

                    q.set_array(swarm_color)
                    ax.plot(sharks_pos[:, 0], sharks_pos[:, 1], sharks_pos[:, 2], 'o', color='#FF0000')
                    for each_food in foods:
                        ax.scatter(each_food.pos[0], each_food.pos[1], each_food.pos[2], color='#228B22')

                    ax.set_xlim(-free_offset, area_width + free_offset)
                    ax.set_ylim(-free_offset, area_height + free_offset)
                    ax.set_zlim(-free_offset, area_depth + free_offset)

                plt.pause(0.00000001)
            for agent in swarm:
                d = [0, 0, 0]
                d_social = [0, 0, 0]
                d_r = [0, 0, 0]
                d_o = [0, 0, 0]
                d_a = [0, 0, 0]
                d_thr = [0, 0, 0]
                d_res = [0, 0, 0]

                if agent.is_alive:
                    for neighbor in swarm:
                        if agent.id != neighbor.id and neighbor.is_alive and norm(neighbor.pos - agent.pos) < r_a:

                            r = neighbor.pos - agent.pos
                            r_normalized = r/norm(r)
                            norm_r = norm(r)
                            agent_vel_normalized = agent.vel/norm(agent.vel)
                            neighbor_vel_normalized = neighbor.vel / norm(neighbor.vel)

                            if acos(np.dot(r_normalized, agent_vel_normalized)) < field_of_view / 2:
                                if norm_r < r_r:
                                    d_r = d_r - r_normalized
                                elif norm_r < r_o:
                                    d_o = d_o + neighbor.vel/norm(neighbor.vel)
                                elif norm_r < r_a:
                                    d_a = d_a + r_normalized
                    if norm(d_r) != 0:
                        d_social = d_r
                    elif norm(d_a) != 0 and norm(d_o) != 0:
                        d_social = (d_o + d_a)/2
                    elif norm(d_a) != 0:
                        d_social = d_o
                    elif norm(d_o) != 0:
                        d_social = d_a

                    # sspacing.append(np.mean(nspacing))  ###average the distance of each agent to each of its neighbors
                    # sdisorder.append(np.mean(ndisorder))  ###average the cohesion of each agent and all its neighbors


                    for each_shark in sharks:
                        if norm(agent.pos - each_shark.pos) <= r_thr:
                            d_thr = d_thr - (each_shark.pos - agent.pos)/norm(each_shark.pos - agent.pos) ** 2
                            if norm(agent.pos - each_shark.pos) <= r_lethal:
                                agent.is_alive = 0
                                number_of_alives = number_of_alives - 1
                                #print('number of alives: ', number_of_alives)

                    for each_food in foods:
                        if norm(agent.pos - each_food.pos) <= r_res:
                            d_res = d_res + (each_food.pos - agent.pos)/norm(each_food.pos - agent.pos) ** 2

                    if norm(d_social) != 0:
                        d = alpha * d_social / norm(d_social)
                    if norm(d_thr) != 0:
                        d = d + (1 - alpha)/2 * d_thr / norm(d_thr)
                    if norm(d_res) != 0:
                        d = d + (1 - alpha)/2 * d_res/norm(d_res)

                    if norm(d) != 0:
                        z = np.cross(d/norm(d), agent.vel/norm(agent.vel))
                        angle_between = asin(norm(z))
                        if angle_between >= theta_dot_max*dt:
                            rot = rotation_matrix_about(z, theta_dot_max*dt)
                            agent.vel = np.asmatrix(agent.vel) * rot
                            agent.vel = np.asarray(agent.vel)[0]
                        elif abs(angle_between)-pi > 0:
                            agent.vel = d/norm(d) * constant_speed


            for each_shark in sharks:
                d = [0, 0, 0]
                dist = 100
                for prey in swarm:
                    r = prey.pos - each_shark.pos
                    r_norm = norm(r)
                    if norm(r) < dist and prey.is_alive and \
                            acos(np.dot(r/r_norm, each_shark.vel/norm(each_shark.vel))) < field_of_view_shark / 2:
                        dist = r_norm
                        d = r / r_norm

                if norm(d) != 0:
                    z = np.cross(d/norm(d), each_shark.vel/norm(each_shark.vel))
                    angle_between = asin(norm(z))
                    if angle_between >= theta_dot_max_shark*dt:
                        rot = rotation_matrix_about(z, theta_dot_max_shark*dt)
                        each_shark.vel = np.asmatrix(each_shark.vel) * rot
                        each_shark.vel = np.asarray(each_shark.vel)[0]
                    elif abs(angle_between)-pi > 0:
                        each_shark.vel = d/norm(d) * shark_speed

            [agent.update_position(dt) for agent in swarm]
            [agent.update_position(dt) for agent in sharks]



        number_of_alives_list.append(number_of_alives)



        print('alpha:', alpha, 'number_of_alives:', number_of_alives,'spacing avg:', total_spacing/max_steps)
        print('time: ', time.time()-start)
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.plot(alpha_list, number_of_alives_list)
    plt.pause(0.0001)



