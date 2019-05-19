import numpy as np
from numpy.linalg import *
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dimension = '2d'    # 2d/3d
n = 40             # number of agents
dt = 0.1
r_r = 1
r_o = 10
r_a = 20
field_of_view = 3*pi/2
theta_dot_max = 1
constant_speed = 2
np.seterr(divide='ignore', invalid='ignore')


class Field:
    def __init__(self):
        self.width = 50    # x_max[m]
        self.height = 50   # y_max[m]
        self.depth = 50    # z_max[m]


class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        self.pos = np.array([0, 0, 0])
        self.pos[0] = np.random.uniform(0, field.width)
        self.pos[1] = np.random.uniform(0, field.height)
        self.pos[2] = np.random.uniform(0, field.depth)
        self.vel = np.random.uniform(-1, 1, 3)
        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0
        self.vel = self.vel / norm(self.vel) * speed

    def update_position(self, delta_t):
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

    swarm = []
    field = Field()
    [swarm.append(Agent(i, constant_speed)) for i in range(n)]

    fig = plt.figure()
    if dimension == '3d':
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    while 1:
        x = np.array([])
        y = np.array([])
        z = np.array([])
        x_dot = np.array([])
        y_dot = np.array([])
        z_dot = np.array([])
        for agent in swarm:
            x = np.append(x, agent.pos[0])
            y = np.append(y, agent.pos[1])
            z = np.append(z, agent.pos[2])
            x_dot = np.append(x_dot, agent.vel[0])
            y_dot = np.append(y_dot, agent.vel[1])
            z_dot = np.append(z_dot, agent.vel[2])

        ax.clear()
        if dimension == '2d':
            ax.quiver(x, y, x_dot / 80 * field.width, y_dot / 80 * field.width, color='#377eb8')
            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, field.width)
            ax.set_ylim(0, field.height)
        else:
            ax.quiver(x, y, z, x_dot / 80 * field.width, y_dot / 80 * field.width,  z_dot / 80 * field.width, color='#377eb8')
            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, field.width)
            ax.set_ylim(0, field.height)
            ax.set_zlim(0, field.depth)
        plt.pause(0.0001)

        for agent in swarm:
            d = 0
            d_r = 0
            d_o = 0
            d_a = 0
            for neighbor in swarm:
                if agent.id != neighbor.id:
                    r = neighbor.pos - agent.pos
                    r_normalized = r/norm(r)
                    norm_r = norm(r)
                    agent_vel_normalized = agent.vel/norm(agent.vel)
                    if acos(np.dot(r_normalized, agent_vel_normalized)) < field_of_view / 2:
                        if norm_r < r_r:
                            d_r = d_r - r_normalized
                        elif norm_r < r_o:
                            d_o = d_o + neighbor.vel/norm(neighbor.vel)
                        elif norm_r < r_a:
                            d_a = d_a + r_normalized
            if norm(d_r) != 0:
                d = d_r
            elif norm(d_a) != 0 and norm(d_o) != 0:
                d = (d_o + d_a)/2
            elif norm(d_a) != 0:
                d = d_o
            elif norm(d_o) != 0:
                d = d_a
            if norm(d) != 0:
                z = np.cross(d/norm(d), agent.vel/norm(agent.vel))
                angle_between = asin(norm(z))
                if angle_between >= theta_dot_max*dt:
                    rot = rotation_matrix_about(z, theta_dot_max*dt)
                    agent.vel = np.asmatrix(agent.vel) * rot
                    agent.vel = np.asarray(agent.vel)[0]
                elif abs(angle_between)-pi > 0:
                    agent.vel = d/norm(d) * 2

        [agent.update_position(dt) for agent in swarm]
