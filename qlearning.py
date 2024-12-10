import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd

import sys

np.random.seed(123)

env = gym.make("CartPole-v0")
env.seed(321)

# Whether to perform training or use the stored .npy file
MODE = "TRAINING"  # TRAINING, TEST
SCHEDULE = "CONSTANT"  # CONSTANT, GLIE, ZERO, ZERO_FIFTY
episodes = 20000
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0.2  # Epsilon for epsilon-greedy policy
if SCHEDULE == "ZERO" or SCHEDULE == "ZERO_FIFTY":
    constant_eps = 0

b = 2221

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
if SCHEDULE == "ZERO_FIFTY":
    # Initialize Q values to 50
    q_grid = np.ones((discr, discr, discr, discr, num_of_actions)) * 50
else:
    q_grid = np.zeros((discr, discr, discr, discr, num_of_actions))

q_grid_initial, q_grid_after_one_episode, q_grid_halfway = (
    None,
    None,
    None,
)


def find_nearest(array, value):
    """
    Find the nearest value in an array
    Args:
        array: array to search
        value: value to find

    Returns:
        idx: index of the nearest value
    """
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    """
    Returns discrete state from continuous state
    Args:
        state: continuous state

    Returns:
        x, v, th, av: discrete state
    """
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, epsilon, num_of_actions=2, greedy=False):
    """
    Returns the action to take in a state using epsilon-greedy policy
    Args:
        state: current state
        q_values: Q-value array
        epsilon: epsilon value for epsilon-greedy policy
        greedy: whether to act greedily

    Returns:
        action: action to take

    """
    x, v, th, av = get_cell_index(state)

    if greedy:  # TEST -> greedy policy
        # TODO: greedy w.r.t. q_grid
        best_action_estimated = np.argmax(q_values[x, v, th, av, :])

        return best_action_estimated

    else:  # TRAINING -> epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Random action
            # TODO: choose random action with equal probability among all actions
            return np.random.choice(num_of_actions)  # Random action
        else:
            # Greedy action
            # TODO: greedy w.r.t. q_grid
            return np.argmax(q_values[x, v, th, av, :])  # Best action estimated


def update_q_value(old_state, action, new_state, reward, done, q_array):
    """
    Update Q-value of the state-action pair using the Bellman equation
    Args:
        old_state: state before taking action
        action: action taken
        new_state: state after taking action
        reward: reward received after taking action
        done: whether the episode is finished
        q_array: Q-value array

    Returns:
        None
    """
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        # TODO: implement the Bellman equation
        target_value = reward + gamma * np.max(
            q_array[
                new_cell_index[0],
                new_cell_index[1],
                new_cell_index[2],
                new_cell_index[3],
                :,
            ]
        )

    # Update Q value
    q_grid[
        old_cell_index[0],
        old_cell_index[1],
        old_cell_index[2],
        old_cell_index[3],
        action,
    ] = alpha * (
        target_value
        - q_array[
            old_cell_index[0],
            old_cell_index[1],
            old_cell_index[2],
            old_cell_index[3],
            action,
        ]
    )

    return


def name_exp(schedule, epsilon=None):
    """
    Get the name of the experiment
    Args:
        schedule: schedule of the experiment

    Returns:
    """
    if schedule == "CONSTANT":
        return "constant" + "_" + str(epsilon)
    if schedule == "GLIE":
        return "GLIE"
    if schedule == "ZERO":
        return "zero_epsilon"
    if schedule == "ZERO_FIFTY":
        return "zero_epsilon_fifty_initial"


def compute_heatmap(q_grid):
    """
    Compute the heatmap of the value function
    Args:
        q_grid: Q-value array

    Returns:
        value_function_2d: 2D value function
    """

    # Compute the value function: max over actions
    value_function = np.max(q_grid, axis=-1)
    # Average over velocity (index 1) and angular velocity (index 3)
    value_function_2d = np.mean(value_function, axis=(1, 3))
    return value_function_2d


def plot_heatmap(heatmap, title, x_min, x_max, y_min, y_max, path=None):
    """
    Plot the heatmap of the value function
    Args:
        heatmap: 2D value function
        title: title of the plot
        x_min: minimum value in x-axis
        x_max: maximum value in x-axis
        v_min: minimum value in y-axis
        v_max: maximum value in y-axis

    """
    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Value Function")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (v)")  # Angle of the pole
    plt.title(title)
    if path is not None:
        plt.savefig(path)
        print("Heatmap saved to", path)
    else:
        plt.show()


def save_model(path, q_values):
    np.save(path, q_values)
    print("Model saved to", path)


model = name_exp(SCHEDULE, constant_eps)

if MODE == "TEST":
    # Check if the Q-value array exists
    try:
        q_grid = np.load("./data/model/q_values_" + model + ".npy")
    except FileNotFoundError:
        print("Q-value file not found. Exiting...")
        sys.exit()

# Training loop
ep_lengths, epl_avg = [], []
std_dev = []
q_grid_initial = q_grid.copy()  # Before training
for ep in range(episodes + test_episodes):
    test = ep > episodes

    if MODE == "TEST":
        test = True

    state, done, steps = env.reset(), False, 0

    epsilon = constant_eps  # TODO: change to GLIE schedule (task 3.1) or 0 (task 3.3)
    if SCHEDULE == "GLIE":
        epsilon = b / (b + ep)
        print("Epsilon:", epsilon)

    total_reward = 0
    while not done:
        action = get_action(state, q_grid, epsilon, greedy=test)
        new_state, reward, done, _ = env.step(action)
        total_reward += reward  # Accumulate reward
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            # print('Test episode:', ep-episodes)
            env.render()

        state = new_state
        steps += 1

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep - 500) :]))
    std_dev.append(np.std(ep_lengths[max(0, ep - 500) :]))
    if ep % 200 == 0:
        print(
            "Episode {}, average timesteps: {:.2f}".format(
                ep, np.mean(ep_lengths[max(0, ep - 200) :])
            )
        )
        print("Epsilon:", epsilon)
        print("Standard deviation:", np.std(ep_lengths[max(0, ep - 200) :]))

    if ep == 1:
        q_grid_after_one_episode = q_grid.copy()  # After one episode
    elif ep == episodes // 2:
        q_grid_halfway = q_grid.copy()  # Halfway through training

if MODE == "TEST":
    sys.exit()

# Save the Q-value array
save_model("./data/model/q_values_" + model + ".npy", q_grid)

save_model("./data/model/q_values_initial_" + model + ".npy", q_grid_initial)

save_model(
    "./data/model/q_values_after_one_episode_" + model + ".npy",
    q_grid_after_one_episode,
)

save_model("./data/model/q_values_halfway_" + model + ".npy", q_grid_halfway)

# Plot the learning curve
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.legend(["Reward Length", "500 Reward Average"])
plt.grid()
plt.xlim(0, episodes + test_episodes)
plt.ylim(0, 210)
# Save the plot
plt.savefig("data/plot/q_learning_" + model + ".png")
print("Plot saved to data/plot/q_learning_" + model + ".png")

# Plot the heatmap for the Q-value array
# --------------------------------------------------------------

# Compute the heatmap of the value function
value_function_2d = compute_heatmap(q_grid)
# Plot the heatmap
plot_heatmap(
    value_function_2d,
    "Value Function",
    x_min,
    x_max,
    th_min,
    th_max,
    path="./data/plot/heatmap_full_training_" + model + ".png",
)


# Compute the heatmap of the value function at the beginning of training
value_function_2d = compute_heatmap(q_grid_initial)
# Plot the heatmap
plot_heatmap(
    value_function_2d,
    "Value Function after One Episode",
    x_min,
    x_max,
    th_min,
    th_max,
    path="./data/plot/heatmap_initial_training_" + model + ".png",
)

# Compute the heatmap of the value function after one episode
value_function_2d = compute_heatmap(q_grid_after_one_episode)
# Plot the heatmap
plot_heatmap(
    value_function_2d,
    "Value Function after One Episode",
    x_min,
    x_max,
    th_min,
    th_max,
    path="./data/plot/heatmap_after_one_episode_" + model + ".png",
)

# Compute the heatmap of the value function halfway through training
value_function_2d = compute_heatmap(q_grid_halfway)
# Plot the heatmap
plot_heatmap(
    value_function_2d,
    "Value Function Halfway Through Training",
    x_min,
    x_max,
    v_min,
    v_max,
    path="./data/plot/heatmap_halfway_" + model + ".png",
)