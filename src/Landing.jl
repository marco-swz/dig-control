using JuliaSimControl.MPC: TerminalInput
using Plots: debug!
using JuliaSimControl
using JuliaSimControl.MPC
using JuliaSimControl.Symbolics
using Interpolations
using StaticArrays
using Plots

# Global variables
const g = 9.81                          # m/s**2

# Rocket dimensions
const LENGTH = 50                       # m
const WIDTH = 9                         # m

# Rocket weight
const MASS_FUEL = 1000000/5                # kg (liquid methane)
const MASS_DRY = 120000                    # kg (rocket body + payload)
const MASS_TOTAL = MASS_DRY + MASS_FUEL

# Raptor v1 engine characteristics (only 3 engines have been seen working so far)
const V_EXHAUST = 3*3280           # m/s exhaust velocity at sea level. 3 sea level + 3 vacuum engines
const THRUST_MAX = 3*2300000/V_EXHAUST  # 701 kg/s. One Raptor thrust is 2.3 MN

# # Inertia for a uniform density rod 
# I = (1/12) * m_total * length^2

# Torque from engines thrust vestoring 
const DEFLECTION_MAX = deg2rad(20)      # thrust vectoring +-20°
const DEFLECTION_MIN = -DEFLECTION_MAX
const TORQUE_MAX = THRUST_MAX * V_EXHAUST * LENGTH/2 * sin(DEFLECTION_MAX)
const TORGUE_MIN = -THRUST_MAX *V_EXHAUST * LENGTH/2 * sin(DEFLECTION_MAX)

# Initial conditions
const X_INIT = -600                  # [m] entry x coordinate
const D_X_INIT = -50                  # [m/s] entry horizontal speed
const Y_INIT = 5000                  # [m] entry altitude
const D_Y_INIT = 0                    # [m/s] entry vertical speed
const ANGLE_INIT = deg2rad(-90)      # [radian]  - rockets starts free falling in belly down position 
const D_ANGLE_INIT = 0                # [radian/s]
const MASS_INIT = MASS_TOTAL               # [kg]
const THRUST_INIT = 0                     # [kg/s] - engines off
const ANGLE_THRUST_INIT = 0          # [radian]
const TORQUE_INIT = 0                # [N*m]
const SAMPLE_TIME = 0.001           # [s]

# Final conditions
const X_FINAL = 0                     # [m] landing x coordinate. Bottom middle of sim box 
const D_X_FINAL = 0                    # [m/s] landing altitude
const Y_FINAL = 0                     # [m] landing altitude
const D_Y_FINAL = 0                    # [m/s] landing speed
const ANGLE_FINAL = 0                 # [radian] - land upright 
const D_ANGLE_FINAL = 0                # [radian/s]
# const m_landing = 0.5*m_total           # [kg]
# const u_landing = 0                     # [kg/s]
const ANGLE_TRHUST_FINAL = 0          # [radian]
const TORQUE_FINAL = 0                # [N*m]

# Number of mesh points (knots) to be used
const N = 100
const NUM_CONTROLS = 2
const NUM_STATES = 9
const OPTIMIZATION_HORIZON = 100

function rocket(state, control, _parameters, _=0)
    # Destructure state and control variables
    y, x, angle, mass, d_y, d_x, d_angle, d_t, fuel = state
    thrust, angle_thrust = control

    torque = -0.5 * LENGTH * V_EXHAUST * thrust * sin(angle_thrust)

    dd_y = (-mass * g + V_EXHAUST * thrust * cos(angle_thrust + angle)) / mass
    dd_x = (V_EXHAUST * thrust * sin(angle_thrust + angle)) / mass
    # ang accel = torque / moment of inertia
    dd_angle = torque / (mass * LENGTH^2 / 12)

    mass -= thrust * d_t

    y += d_y * d_t
    x += d_x * d_t
    angle += d_angle * d_t

    d_x += dd_x * d_t
    d_y += dd_y * d_t
    d_angle += dd_angle * d_t

    fuel += thrust

    return SA[y, x, angle, mass, d_y, d_x, d_angle, d_t, fuel]
end

state_init = Float64[Y_INIT, X_INIT, ANGLE_INIT, MASS_INIT, D_Y_INIT, D_X_INIT, D_ANGLE_INIT, SAMPLE_TIME, 0]
state_final = Float64[Y_FINAL, X_FINAL, ANGLE_FINAL, MASS_DRY, D_Y_FINAL, D_X_FINAL, D_ANGLE_FINAL, SAMPLE_TIME, 0]

# The entire state is available for measurement
measurement = (x, u, p, t) -> x 

dynamics = FunctionSystem(
    rocket, 
    measurement;
    x=[:y, :x, :angle, :mass, :d_y, :d_x, :d_angle, :d_t, :fuel], 
    u=[:thrust, :angle_thrust], 
    y=:y^NUM_STATES
)
discrete_dynamics = MPC.rk4(dynamics, SAMPLE_TIME; supersample=3)

upper_bounds = [Y_INIT, 1500, 2*pi, MASS_TOTAL, 0, 80, deg2rad(45), SAMPLE_TIME, THRUST_MAX, DEFLECTION_MAX]
lower_bounds = [0, -1500, -2*pi, MASS_DRY, -80, -80, -deg2rad(45), SAMPLE_TIME, 0, DEFLECTION_MIN]

# Add lower and upper bounds
stage_constraint = StageConstraint(lower_bounds, upper_bounds) do si, p, t
    # NOTE: The re-formating is required to align the structure of the state
    # with the format of the constraints.
    y, x, angle, mass, d_y, d_x, d_angle, d_t, fuel = si.x
    thrust, angle_thrust = si.u
    return SA[y, x, angle, mass, d_y, d_x, d_angle, d_t, thrust, angle_thrust]
end

# Add a terminal constraint for the final mass (it must be `mass_dry`)
terminal_constraint = TerminalStateConstraint([MASS_DRY], [MASS_DRY]) do ti, p, t
    mass = ti.x[4]
    return SA[mass]
end

# Define an observer
observer = StateFeedback(discrete_dynamics, state_init)

# Define the loss and objective function
loss = TerminalCost() do ti, p, t
    x = ti.x[2]
    d_x = ti.x[6]
    thrust = ti.x[9]
    return sum(1*d_x.^2 + 1*thrust.^2 + 1*x.^2)
end
objective = Objective(loss)

# Create objective input
reference = zeros(NUM_STATES)
thrust = zeros(NUM_CONTROLS, OPTIMIZATION_HORIZON)
state_trajectory, thrust_trajectory = MPC.rollout(discrete_dynamics, state_init, thrust, 0, 0)
objective_input = ObjectiveInput(state_trajectory, thrust_trajectory, reference)

# Define the solver
solver = IpoptSolver(;
    verbose=false,
    tol=1e-5,
    acceptable_tol=1e-5,
    constr_viol_tol=1e-5,
    acceptable_constr_viol_tol=1e-5,
    acceptable_iter=50,
    exact_hessian=false,
)

# Define the nonlinear Model-Predictive Control problem
problem = GenericMPCProblem(
    dynamics;
    N=OPTIMIZATION_HORIZON,
    observer=observer,
    Ts=SAMPLE_TIME,
    objective=objective,
    solver=solver,
    constraints=[stage_constraint, terminal_constraint],
    objective_input=objective_input,
    xr=reference,
    presolve=true,
    verbose=true,
    jacobian_method=:forwarddiff, # generation of symbolic constraint Jacobians and Hessians are beneficial when using Trapezoidal as discretization.
    gradient_method=:reversediff,
    hessian_method=:none,
    disc=Trapezoidal(; dyn=dynamics),
)

x_sol, u_sol = get_xu(problem)
#plot(x_sol[2, :], x_sol[1, :])
plot(
    plot(x_sol', title="States", lab=permutedims(state_names(dynamics)), layout=(3, 3)),
    #plot(u_sol', title="Control signal", lab=permutedims(input_names(dynamics))),
)

