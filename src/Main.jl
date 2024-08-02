using JuliaSimControl
using JuliaSimControl.MPC
using JuliaSimControl.Symbolics
using StaticArrays
using Plots

const height_initial = 1
const velocity_initial = 0
const mass_initial = 1
const gravity_initial = 1

const T_c = 3.5  # Used for thrust
const mass_reduction = 0.6 

const drag_height_factor = 500
const drag_vel_factor = 620

mass_final = mass_reduction * mass_initial
thrust_max = T_c * gravity_initial * mass_initial


function rocket(state, control, _parameters, _=0)
    mass_loss_factor= 0.5 * sqrt(gravity_initial * height_initial)
    drag_scaling = 0.5 * mass_initial / gravity_initial

    height, velocity, mass = state
    drag = drag_scaling * drag_vel_factor * velocity^2 * exp(-drag_height_factor * (height - height_initial) / height_initial)
    gravity = gravity_initial * (height_initial / height)^2
    thrust = control[]

    # Return the dynamics of the system
    return SA[
        velocity
        (thrust - drag - mass * gravity) / mass
        -thrust / mass_loss_factor
    ]
end


num_controls = 1
num_states = 3
num_steps = 200 # Optimization horizon (number of time steps)
sample_time = 0.001
state_initial = Float64[height_initial, velocity_initial, mass_initial]

# The entire state is available for measurement
measurement = (x, u, p, t) -> x 

dynamics = FunctionSystem(rocket, measurement; x=[:height, :velocity, :mass], u=:thrust, y=:y^num_states)
discrete_dynamics = MPC.rk4(dynamics, sample_time; supersample=3)

lower_bounds = [height_initial, velocity_initial, mass_final, 0]
upper_bounds = [Inf, Inf, mass_initial, thrust_max]

# Add lower and upper bounds
stage_constraint = StageConstraint(lower_bounds, upper_bounds) do si, p, t
    # NOTE: The re-formating is required to align the structure of the state
    # with the format of the constraints.
    height, velocity, mass = si.x
    thrust = (si.u)[]
    return SA[height, velocity, mass, thrust]
end

# Add a terminal constraint for the final mass (it must be `mass_final`)
terminal_constraint = TerminalStateConstraint([mass_final], [mass_final]) do ti, p, t
    mass = ti.x[3]
    return SA[mass]
end

# Simulate the discrete system with manual thrust inputs
thrust = [0.7thrust_max * ones(num_controls, num_steps ÷ 5) thrust_max / 5 * ones(num_controls, 4num_steps ÷ 5)]
state_trajectory, thrust_trajectory = MPC.rollout(discrete_dynamics, state_initial, thrust, 0, 0)

plot(state_trajectory', layout=3)

exit()

# Extract the initial guesses for state and control trajectories
reference = zeros(num_states)
objective_input = ObjectiveInput(state_trajectory, thrust_trajectory, reference)

# Define an observer
observer = StateFeedback(discrete_dynamics, state_initial)

# Define the loss and objective function
loss = TerminalCost() do ti, p, t
    height = ti.x[1]
    -height # Maximize the terminal altitude
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
