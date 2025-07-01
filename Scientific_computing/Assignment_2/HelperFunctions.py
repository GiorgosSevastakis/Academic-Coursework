import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, N, x_range, y_range, dt, Nt, C, Dp, Dq):
        self.N = N
        self.x = np.linspace(*x_range, N)
        self.y = np.linspace(*y_range, N)
        self.dx = self.x[2] - self.x[1]
        self.dy = self.y[2] - self.y[1]
        self.dt = dt
        self.T = np.zeros(Nt)
        self.C = C
        self.Dp = Dp
        self.Dq = Dq

    def initial_conditions(self, K):
        P = np.zeros((self.N, self.N))
        Q = np.zeros((self.N, self.N))
        condition_X = (self.x > 10) & (self.x < 30)
        condition_Y = (self.y > 10) & (self.y < 30)
        mask = np.outer(condition_X, condition_Y)
        P[mask] = self.C + 0.1
        Q[mask] = (K / self.C) + 0.2
        return P, Q

    def laplacian(self, M):
        CM = np.copy(M)
        CM = (np.roll(CM, -1, axis=0) + np.roll(CM, 1, axis=0) - 2 * CM) / (self.dx**2) + \
             (np.roll(CM, -1, axis=1) + np.roll(CM, 1, axis=1) - 2 * CM) / (self.dy**2)
        return CM

    def euler(self, K):
        p, q = self.initial_conditions(K)
        for i in range(len(self.T)):
            p += (self.Dp * self.laplacian(p) + (p * p) * q + self.C - (K + 1) * p) * self.dt
            q += (self.Dq * self.laplacian(q) - (p * p) * q + K * p) * self.dt
            p[0, :], p[-1, :], p[:, 0], p[:, -1] = p[2, :], p[-3, :], p[:, 2], p[:, -3]
            q[0, :], q[-1, :], q[:, 0], q[:, -1] = q[2, :], q[-3, :], q[:, 2], q[:, -3]
        return p, q

    def plot_results(self, K, pd, qd):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        first = ax1.contourf(pd, 30, cmap='ocean')
        second = ax2.contourf(qd, 30, cmap='ocean')
        ax1.set_xlabel('y', fontsize = 13)
        ax1.set_ylabel('x', fontsize = 13)
        ax2.set_xlabel('y', fontsize = 13)
        ax2.set_ylabel('x', fontsize = 13)
        ax1.set_title(f'p(x,y,t=2000) for K={K}', fontsize = 16)
        ax2.set_title(f'q(x,y,t=2000) for K={K}', fontsize = 16)
        plt.colorbar(first, ax=ax1)
        plt.colorbar(second, ax=ax2)
        fig.tight_layout()
        plt.show()
