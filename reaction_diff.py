import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

class ReactionDiff:
    """
    Brief
    ------
    Class for simulating a system of 2 reaction-diffusion models via the Crank-Nicolson method.
        u_t = D_u * u_xx + f_u
        v_t = D_v * v_xx + f_v
    U and V denote the 2 concentration/reaction-diffusion model variables that have to be approximated.

    Key Attributes/Variables
    -------------------------
    U - the approximated solution for the U variable, an X by T matrix
    V - the approximated solution for the V ariable, an X by T matrix

    Methods
    --------
    __init__(...)
        Initializes important variables for the simulation: spatio-temporal variables, diffusion speed and others.
    
    set_initial_conditions(...)
        Sets the initial conditions for the U and V variables
    
    register_u_ODE_func(...)
        Registers the reaction/ODE part of the reaction-diffusion model for U (f_u). 
    
    register_v_ODE_func(...)
        Registers the reaction/ODE part of the reaction-diffusion model for V (f_v).

    Solve(...)
        Numerically solves for the system.
    """

    def __init__(self, x0, x_max, dx, t0, t_max, dt, Du=1, Dv=1):
        """
        Initializes the reaction-diffusion model.

        Params:
            x0 - the left spatial boundary
            x_max - the right spatial boundary
            dx - spatial increment
            t0 - the beggining time
            t_max - the edning time
            dt - temporal increment
            Du - diffusion speed of U
            Dv - diffusion speed of V
        
        """
        self.x0 = x0
        self.x_max = x_max
        self.dx = dx
        self.t0 = t0
        self.t_max = t_max
        self.dt = dt
        self.Du = Du
        self.Dv = Dv

        #Setting up spatial variables
        self.x_vals = np.linspace(x0, x_max, int((x_max-x0)/dx))
        self.N = len(self.x_vals)

        #Setting up temporal variables
        self.t_vals = np.linspace(t0, t_max, num=int((t_max-t0)/dt))
        self.T = len(self.t_vals)
        
        #Setting up the A matrix for the finite-difference method
        F = np.ones(self.N)
        diag_data = np.array((F, -2*F, F))
        diags = np.array([-1, 0,1])
        A = scipy.sparse.spdiags(diag_data,diags, self.N, self.N).toarray()
        A[-1][-1] = -2
        A[-1][-2] = -2

        # Setting up the coefficient matrices for u and v variables
        cu = (Du*dt)/(2*dx**2)
        cv = (Dv*dt)/(2*dx**2)
        
        diag_data = np.array((F))
        diags = np.array([0])
        I = scipy.sparse.spdiags(diag_data,diags, self.N, self.N).toarray()

        self.C1u = I - cu*A
        self.C2u = I + cu*A
        self.C1v = I - cv*A
        self.C2v = I + cv*A

        #Default boundary conditions
        self.Bu = np.zeros(self.N)
        self.Bv = np.zeros(self.N)

        au = 0.5
        bu = 0

        av = 0
        bv = 0
        self.Bu[0] = au*2*cu
        self.Bu[-1] = bu*2*cu
        self.Bv[0] = av*2*cv
        self.Bv[-1] = bv*2*cv

        #Setting up the matrices which hold the solutions
        self.U = np.zeros((self.N,self.T))
        self.V = np.zeros((self.N,self.T))
    
    def set_boundary_conditions(self,Bu, Bv):
        return 
    
    def set_initial_conditions(self,u0, v0):
        self.u_0 = u0
        self.v_0 = v0
        return
    
    def register_u_ODE_func(self, u_func):
        """
        Registers the reaction/ODE part of the model

        Params:
            u_func - lambda function of 2 variables (u, v)
        """
        self.u_f = u_func
    
    def register_v_ODE_func(self, v_func):
        """
        Registers the reaction/ODE part of the model

        Params:
            v_func - lambda function of 2 variables (u, v)
        """
        self.v_f = v_func
    
    def Solve(self, animate=True):
        """
        Numerically solves for the reaction-diffusion model using the Crank-Nicolson Method

        Params:
            animate - if True, will show animations of the solution and the final concentration gradients for U and V
        """
        self.U[:,0] = self.u_0
        self.V[:,0] = self.v_0
        u = np.asarray(self.u_0)
        v = np.asarray(self.v_0)

        if animate:
            fig, axs = plt.subplots(nrows=1)
            axs.set_ylim([0,1.1])
            axs.set_title("BMP and NODAL Waves for $\gamma$ = 1.08")
            axs.set_xlabel("X (distance from edge of colony to the center)")
            axs.set_ylabel("Concentration of U and V")
            line1 = axs.plot(self.x_vals, self.u_0, label="U (BMP)")
            line1[0].set_label("u (BMP)")
            line2 = axs.plot(self.x_vals, self.v_0, label="V (NODAL)")
            line2[0].set_label("v (NODAL)")
            plt.legend(handles=[line1[0], line2[0]])
            # plt.show()
            plt.ion()
        
        for t in range(0,self.T):    
            if t != 0:
                u_new = self.C2u@u + self.u_f(u,v) + self.Bu
                u_new = np.linalg.solve(self.C1u, u_new)
                self.U[:, t] = u_new
                u = u_new

                v_new = self.C2v@v + self.v_f(u, v) + self.Bv
                v_new = np.linalg.solve(self.C1v, v_new)
                self.V[:, t] = v_new
                v = v_new
            if t % 20 == 0 and animate:
                line1[0].set_xdata(self.x_vals)
                line1[0].set_ydata(self.U[:,t])
                line2[0].set_xdata(self.x_vals)
                line2[0].set_ydata(self.V[:,t])
                fig.canvas.draw()
                # fig.legend()
                fig.canvas.flush_events()
                plt.pause(0.1)
        
        if animate:
            # plt.show()
            plt.ioff()
            fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
            X,Y = np.meshgrid(self.t_vals,self.x_vals)
            bmp_im = axs[0].pcolormesh(X, Y, self.U, vmin=0, vmax=1, shading="auto", cmap="viridis")
            axs[0].set_title("BMP (u) Morphogen Concentration")
            axs[0].set_xlabel("Time")
            axs[0].set_ylabel("Distance from edge of Colony/Strip")
            fig.colorbar(bmp_im, ax=axs[0])

            nodal_im = axs[1].pcolormesh(X, Y, self.V, vmin=0, vmax=1, shading="auto", cmap="viridis")
            fig.colorbar(nodal_im, ax=axs[1])
            plt.title("Nodal (v) Morphogen Concentration")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("Distance from edge of Colony/Strip")
            plt.show()

if __name__ == "__main__":
    x_max = 20
    dx = 0.05
    x0 = 0

    dt = 0.01
    t0 = 0
    t_max = 1
    r_diff_model = ReactionDiff(x0,x_max,dx,t0,t_max,dt, Du=1.5, Dv=1)

    N = r_diff_model.N
    # u_0 = np.zeros(N, dtype=np.double)
    # for i in range(len(u_0)):
    #     j = i/N
    #     if j < 0.1:
    #         u_0[i] = 0.4
    #     elif j > 0.4 and j < 0.6:
    #         u_0[i] = 0.4

    # v_0 = np.zeros(N, dtype=np.double)
    # for i in range(len(u_0)):
    #     j = i /N
    #     if j > 0.1 and j < 0.4:
    #         v_0[i] = 0.4
    #     elif j > 0.6 and j < 0.8:
    #         v_0[i] = 0.4

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


    fig, axs = plt.subplots(nrows=1)
    x_vals = r_diff_model.x_vals
    t_vals = r_diff_model.t_vals
    U = r_diff_model.U
    V = r_diff_model.V
    threshold = 0.4
    threshold_line = [threshold for i in range(0, len(x_vals))]

    line1 = axs.plot(x_vals, U[:,5], color="palegreen")
    line1[0].set_label("BMP @ t1")
    line2 = axs.plot(x_vals, U[:, 10], color="lightgreen")
    line2[0].set_label("BMP @ t2")
    line3 = axs.plot(x_vals, U[:,25], color="forestgreen")
    line3[0].set_label("BMP @ t3")
    line4 = axs.plot(x_vals, V[:,25], color="blue")
    line4[0].set_label("NODAL @ t3")
    line5 = axs.plot(x_vals, V[:,10], color="lightblue")
    line5[0].set_label("NODAL @ t2")

    plt.legend(handles=[line1[0], line2[0], line3[0], line5[0], line4[0]])
    axs.set_xlim([0,5])
    axs.set_xlabel("X")
    axs.set_ylabel("U")

    # line2[0].set_label("v (NODAL)")
    axs.plot(x_vals, threshold_line, color="red", label="0.4 Threshold")
    plt.title("BMP Wave Front at Different Time Stamps")
    plt.show()