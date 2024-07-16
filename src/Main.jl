using JuliaSimControl
using JuliaSimControl.MPC
using JuliaSimControl.Symbolics
using LinearAlgebra
using StaticArrays
using Plots

const h_0 = 1    # Initial height
const v_0 = 0    # Initial velocity
const m_0 = 1    # Initial mass
const g_0 = 1    # Gravity at the surface

const T_c = 3.5  # Used for thrust
const m_c = 0.6  # Fraction of initial mass left at end

const m_f = m_c * m_0              # Final mass
const T_max = T_c * g_0 * m_0      # Maximum thrust


function rocket(x, u, p, _=0)
    h_c = 500                    # Used for drag
    v_c = 620                    # Used for drag
    c = 0.5 * sqrt(g_0 * h_0)    # Thrust-to-fuel mass
    D_c = 0.5 * v_c * m_0 / g_0  # Drag scaling

    h, v, m = x
    T = u[]                      # Thrust (control signal)
    drag =  D_c * v^2 * exp(-h_c * (h - h_0) / h_0)
    grav = g_0 * (h_0 / h)^2
    SA[
        v
        (T - drag - m * grav) / m
        -T/c
    ]
end

nu = 1            # number of control inputs
nx = 3            # number of states
N  = 200          # Optimization horizon (number of time steps)
Ts = 0.001        # sample time
x0 = Float64[h_0, v_0, m_0]   # Initial state
r = zeros(nx)

measurement = (x,u,p,t) -> x                # The entire state is available for measurement
dynamics = FunctionSystem(rocket, measurement; x=[:h, :v, :m], u=:T, y=:y^nx)
discrete_dynamics = MPC.rk4(dynamics, Ts; supersample=3)

lb = [h_0, v_0, m_f, 0]
ub = [Inf, Inf, m_0, T_max]

stage_constraint = StageConstraint(lb, ub) do si, p, t
    u = (si.u)[]
    h,v,m = si.x
    SA[h, v, m, u]
end

terminal_constraint = TerminalStateConstraint([m_f], [m_f]) do ti, p, t
    SA[ti.x[3]] # The final mass must be m_f
end

terminal_cost = TerminalCost() do ti, p, t
    h = ti.x[1]
    -h # Maximize the terminal altitude
end

objective = Objective(terminal_cost)

u = [0.7T_max * ones(nu, N÷5)  T_max / 5 * ones(nu, 4N÷5) ]

x, u = MPC.rollout(discrete_dynamics, x0, u, 0, 0)
oi = ObjectiveInput(x, u, r)
plot(x', layout=3)

observer = StateFeedback(discrete_dynamics, x0)

solver = IpoptSolver(;
        verbose                    = false,
        tol                        = 1e-8,
        acceptable_tol             = 1e-5,
        constr_viol_tol            = 1e-8,
        acceptable_constr_viol_tol = 1e-5,
        acceptable_iter            = 10,
)

prob = GenericMPCProblem(
    dynamics;
    N,
    observer,
    Ts,
    objective,
    solver,
    constraints     = [stage_constraint, terminal_constraint],
    objective_input = oi,
    xr              = r,
    presolve        = true,
    verbose         = false,
    jacobian_method = :symbolics, # generation of symbolic constraint Jacobians and Hessians are beneficial when using Trapezoidal as discretization.
    disc  = Trapezoidal(; dyn=dynamics),
)

x_sol, u_sol = get_xu(prob)
plot(
    plot(x_sol', title="States",         lab=permutedims(state_names(dynamics)), layout=(nx,1)),
    plot(u_sol', title="Control signal", lab=permutedims(input_names(dynamics))),
)
