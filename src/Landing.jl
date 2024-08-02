using JuliaSimControl
using JuliaSimControl.MPC
using JuliaSimControl.Symbolics
using Interpolations
using StaticArrays
using Plots

# Global variables
const g = 9.81                          # m/s**2

# Rocket dimensions
const length = 50                       # m
const width = 9                         # m

# Rocket weight
const mass_fuel = 1000000/5                # kg (liquid methane)
const mass_dry = 120000                    # kg (rocket body + payload)
const mass_total = mass_dry + mass_fuel

# Raptor v1 engine characteristics (only 3 engines have been seen working so far)
const v_exhaust = 3*3280           # m/s exhaust velocity at sea level. 3 sea level + 3 vacuum engines
const thrust_max = 3*2300000/v_exhaust  # 701 kg/s. One Raptor thrust is 2.3 MN

# # Inertia for a uniform density rod 
# I = (1/12) * m_total * length^2

# Torque from engines thrust vestoring 
const deflection_max = deg2rad(20)      # thrust vectoring +-20°
const deflection_min = -deflection_max
const torque_max = u_max * v_exhaust * length/2 * sin(deflection_max)
const torque_min = -u_max *v_exhaust * length/2 * sin(deflection_max)

# Initial conditions
const x_init = -600                  # [m] entry x coordinate
const d_x_init = -50                  # [m/s] entry horizontal speed
const y_init = 5000                  # [m] entry altitude
const d_y_init = 0                    # [m/s] entry vertical speed
const angle_init = deg2rad(-90)      # [radian]  - rockets starts free falling in belly down position 
const d_angle_init = 0                # [radian/s]
const mass_init = mass_total               # [kg]
const thrust_init = 0                     # [kg/s] - engines off
const angle_thrust_init = 0          # [radian]
const torque_init = 0                # [N*m]
const sample_time = 0.001           # [s]

# Final conditions
const x_final = 0                     # [m] landing x coordinate. Bottom middle of sim box 
const d_x_final = 0                    # [m/s] landing altitude
const y_final = 0                     # [m] landing altitude
const d_y_final = 0                    # [m/s] landing speed
const angle_final = 0                 # [radian] - land upright 
const d_angle_final = 0                # [radian/s]
# const m_landing = 0.5*m_total           # [kg]
# const u_landing = 0                     # [kg/s]
const angle_thrust_final = 0          # [radian]
const torque_final = 0                # [N*m]

# Number of mesh points (knots) to be used
const n = 100
const num_controls = 1
const num_states = 3
const num_steps = 200 # Optimization horizon (number of time steps)

function rocket(state, control, _parameters, _=0)
    # Destructure state and control variables
    y, x, angle, mass, d_y, d_x, d_angle, d_t = state
    thrust, angle_thrust = control

    torque = -0.5 * length * v_exhaust * thrust * sin(angle_thrust)

    dd_y = (mass * g + v_exhaust * thrust * cos(angle_thrust + angle)) / mass
    dd_x = (v_exhaust * thrust * sin(angle_thrust + angle)) / mass
    # ang accel = torque / moment of inertia
    dd_angle = torque / (mass * length^2 / 12)

    mass = mass - thrust * d_t

    y = y + d_y * d_t
    x = x + d_x * d_t
    angle = angle + d_angle * d_t

    d_x = d_x + dd_x * d_t
    d_y = d_y + dd_y * d_t
    d_angle = d_angle + dd_angle * d_t

    return SA[y, x, mass, d_y, d_x, d_angle, d_t]
end

state_init = Float64[y_init, x_init, mass_init, d_y_init, d_x_init, d_angle_init, sample_time]
state_final = Float64[y_final, x_final, mass_dry, d_y_final, d_x_final, d_angle_final, sample_time]

# The entire state is available for measurement
measurement = (x, u, p, t) -> x 

dynamics = FunctionSystem(
    rocket, 
    measurement; 
    x=[:y, :x, :mass, :d_y, :d_x, :d_angle, :d_t], 
    u=[:thrust, :angle_thrust], y=:y^num_states
)
discrete_dynamics = MPC.rk4(dynamics, sample_time; supersample=3)

lower_bounds = [0, -1500, -2*pi, mass_total, -80, -80, -deg2rad(45), sample_time, 0, deflection_min]
upper_bounds = [y_init, 1500, 2*pi, mass_dry, 0, 80, deg2rad(45), sample_time, thrust_max, deflection_max]

# Add lower and upper bounds
stage_constraint = StageConstraint(lower_bounds, upper_bounds) do si, p, t
    # NOTE: The re-formating is required to align the structure of the state
    # with the format of the constraints.
    y, x, angle, mass, d_y, d_x, d_angle, d_t = si.x
    thrust, angle_thrust = si.u
    return SA[y, x, angle, mass, d_y, d_x, d_angle, thrust, angle_thrust]
end

# Add a terminal constraint for the final mass (it must be `mass_dry`)
terminal_constraint = TerminalStateConstraint([mass_dry], [mass_dry]) do ti, p, t
    mass = ti.x[4]
    return SA[mass]
end

interp_linear = Interpolations.LinearInterpolation([1, n], [state_init, state_final])
initial_guess = Interpolations.mapreduce(transpose, vcat, interp_linear.(1:n))

# Extract the initial guesses for state and control trajectories
reference = zeros(num_states)
objective_input = ObjectiveInput(state_trajectory, thrust_trajectory, reference)

# Define an observer
observer = StateFeedback(discrete_dynamics, state_initial)

# Define the loss and objective function
loss = TerminalCost() do ti, p, t
    x = ti.x[2]
    d_x = ti.x[6]
    thrust = ti.u[1]
    return sum(1*d_x.^2 + 1*thrust.^2 + 1*x.^2)
end
objective = Objective(loss)

# Define the solver
solver = IpoptSolver(;
    verbose=false,
    tol=1e-8,
    acceptable_tol=1e-5,
    constr_viol_tol=1e-8,
    acceptable_constr_viol_tol=1e-5,
    acceptable_iter=10,
)

# Define the nonlinear Model-Predictive Control problem
problem = GenericMPCProblem(
    dynamics;
    num_steps,
    observer,
    sample_time,
    objective,
    solver,
    constraints=[stage_constraint, terminal_constraint],
    objective_input=objective_input,
    xr=reference,
    presolve=true,
    verbose=false,
    jacobian_method=:symbolics, # generation of symbolic constraint Jacobians and Hessians are beneficial when using Trapezoidal as discretization.
    disc=Trapezoidal(; dyn=dynamics),
)

x_sol, u_sol = get_xu(problem)
plot(
    plot(x_sol', title="States", lab=permutedims(state_names(dynamics)), layout=(num_states, 1)),
    plot(u_sol', title="Control signal", lab=permutedims(input_names(dynamics))),
)



print(size(initial_guess))

JuMP.set_start_value.(JuMP.all_variables(model), vec(initial_guess));    
