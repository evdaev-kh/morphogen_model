import numpy as np
import reaction_diff
import matplotlib.pyplot as plt

def get_last_thereshold_value_index(E, th):
    """"
    Brief
    ------
    Finds the index of the last value u s.th u < th

    Params
    -------
    E - an X by T matrix representing solutions for a PDE for a given time
    th - threshold value for the concentration

    Return
    -------
    A 1 x T array of indices satisfying the threshold condition for each time unit
    """

    T = len(E[0])
    W = np.zeros(T)
    for t in range(0, T):
        k = np.argwhere(th > E[:, t])
        if k.size:
            W[t] = k[0][0]
        else:
            W[t] = 0
        # print(k)
        # print(E[k[0][0], t])
        # print(E[k[0][0] - 1, t])
    return W

def calculate_invasion_wave_speed(E, W, x_vals, th, dt):
    """
    Brief
    ------
    Calculates the wave speed for a given index/spatial value

    Params
    -------
    E       - an X by T matrix representing solutions for a PDE for a given time
    W       - a 1 by T matrix representing the last index for each time variable s.th
    u < threshold at a time stamp 
    x_vals  - the spatial values of a PDE
    th      - threshold value for the concentration
    dt      - time delta for each grid/mesh point

    Return
    -------
    S - a 1 by T matrix representing the velocities for the threshold for each time
    """
    N = len(W)
    S = np.zeros(N)
    X = np.zeros(N)

    for t in range(0, N):
        index = int(W[t])
        speed = float()
        if index == len(x):
            speed = 0.0
        elif index == 0:
            speed = 0.0
        else:
            du = E[index, t] - E[index-1, t]
            dx = x_vals[index] - x[index-1]
            speed = du/dx
            x_s = float()
            if speed == 0:
                x_s = x_vals[index-1]
            else:
                x_s = x_vals[index-1] + (th - E[index-1, t])/speed
            X[t] = x_s
        
        if t != 0:
            S[t] = (X[t] - X[t-1])/dt
    
    return S


if __name__ == "__main__":
    x_max = 20
    dx = 0.05
    x0 = 0

    dt = 0.01
    t0 = 0
    t_max = 10
    r_diff_model = reaction_diff.ReactionDiff(x0,x_max,dx,t0,t_max,dt, Du=1.5, Dv=1)

    N = r_diff_model.N
    u_0 = np.zeros(N, dtype=np.double)
    for i in range(len(u_0)):
        j = i/N
        if j < 0.1:
            u_0[i] = 0.4

    v_0 = np.zeros(N, dtype=np.double)
    for i in range(len(u_0)):
        j = i /N
        if j < 0.1 and j < 0.4:
            v_0[i] = 0.01
    
    r_diff_model.set_initial_conditions(u_0, v_0)
    # Define the ODE part of the reaction-diffusion PDE
    gamma = 1.08
    u_ODE = lambda u, v: u*(1-u)*(u-v)
    v_ODE = lambda u, v: v*(1-v-gamma*u)

    r_diff_model.register_u_ODE_func(u_ODE)
    r_diff_model.register_v_ODE_func(v_ODE)
    r_diff_model.Solve(animate=False)
    print("solved")
    W = get_last_thereshold_value_index(r_diff_model.U, 0, 0.4)

    S = calculate_invasion_wave_speed(r_diff_model.U, W, r_diff_model.x_vals, 0.4, dt)
    plt.plot(r_diff_model.t_vals, S)
    print(f"Final Speed={S[-1]}")
    plt.show()



