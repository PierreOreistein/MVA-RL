from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import time

# Graphic packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

env = GridWorld1

################################################################################
# investigate the structure of the environment
# - env.n_states: the number of states
# - env.state2coord: converts state number to coordinates (row, col)
# - env.coord2state: converts coordinates (row, col) into state number
# - env.action_names: converts action number [0,3] into a named action
# - env.state_actions: for each state stores the action availables
#   For example
#       print(env.state_actions[4]) -> [1,3]
#       print(env.action_names[env.state_actions[4]]) -> ['down' 'up']
# - env.gamma: discount factor
################################################################################
print(env.state2coord)
print(env.coord2state)
print(env.state_actions)
for i, el in enumerate(env.state_actions):
    print("s{}: {}".format(i, env.action_names[el]))

################################################################################
# Policy definition
# If you want to represent deterministic action you can just use the number of
# the action. Recall that in the terminal states only action 0 (right) is
# defined.
# In this case, you can use gui.renderpol to visualize the policy
################################################################################

# Deterministioc policy
pol = []
for i, actions in enumerate(env.state_actions):

    # Explore all the actions
    nb_actions = len(actions)
    i = 0
    while i < nb_actions:
        if 'right' == env.action_names[actions[i]]:
            pol.append(actions[i])
            i = nb_actions + 1
        i += 1
    if i == nb_actions:
        pol.append(3)  # Corresponding to action "up"

# Display the policy
# gui.render_policy(env, pol)

################################################################################
# Try to simulate a trajectory
# you can use env.step(s,a, render=True) to visualize the transition
################################################################################
env.render = False
state = 0
fps = 1

for i in range(5):
    action = pol[state]  # np.random.choice(env.state_actions[state])
    nexts, reward, term = env.step(state, action)
    state = nexts
    time.sleep(1./fps)

################################################################################
# You can also visualize the q-function using render_q
################################################################################
# first get the maximum number of actions available
max_act = max(map(len, env.state_actions))
q = np.random.rand(env.n_states, max_act)
# gui.render_q(env, q)

################################################################################
# Work to do: Q4
################################################################################
# here the v-function and q-function to be used for question 4
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]

# Here is defined the deterministioc policy
pol = []
for i, actions in enumerate(env.state_actions):

    # Explore all the actions
    nb_actions = len(actions)
    i = 0
    while i < nb_actions:
        if 'right' == env.action_names[actions[i]]:
            pol.append(actions[i])
            i = nb_actions + 1
        i += 1
    if i == nb_actions:
        pol.append(3)  # Corresponding to action "up"

# Display the policy
# gui.render_policy(env, pol)


# Here the Monte-Carlo Estimation
def estimatorMCValueFunctions(pi, n=1000, gamma=0.95, delta=0.001):
    """Estimate the value functions thanks to a MC estimation."""

    # Definition of Tmax
    Tmax = -np.log(delta / 1) / (1 - gamma)

    # Saving array of all the trajectories starting with s
    tau = [[] for i in range(env.n_states)]

    # Play the different trajectories
    for e in range(n):

        # Initialisation of the first state, the time and the terminal flag
        state = int(env.reset())
        t = 1
        term_flag = False

        # Saving array of all the rewards and and save the first state
        rewards = []
        state_init = state

        while (t < Tmax and not(term_flag)):

            # Select the action to do according to the policy
            action = pol[state]

            # Let act the environment
            state, reward, term_flag = env.step(state, action)

            # TO UNCOMMENT IF WE DO NO WANT TO CONSIDER THE FINAL STATES
            # term_flag=False

            # Add the reward to the saving array
            rewards.append(reward)

            # Increase the timer
            t += 1

        # Update tau
        tau[state_init].append(rewards)

    # Compute the estimation of the value functions
    V_pi = np.zeros((env.n_states, 1))

    for state in range(env.n_states):

        # Extract N(s)
        N_s = len(tau[state])

        # Compute the mean over all the trajectories begin with state
        for trajectory in tau[state]:

            # Pounded mean of this trajectory
            mean = 0
            for t in range(len(trajectory)):
                mean += gamma**t * trajectory[t]

            # Update of V_pi[state]
            V_pi[state, 0] += mean / N_s

    return V_pi

# V_pi = estimatorMCValueFunctions(pol, n=1000)
# print(V_pi)


# Here the function for estimating mu_0
def estimationOfMu0(nb_samples=10000):
    """Compute an MC estimation of mu_0 with nb_samples samples."""

    # Initialisation of mu_0
    mu_0 = np.zeros((env.n_states, 1))

    # Compute the estimation
    for sample in range(nb_samples):

        # Sample of mu_0
        state = int(env.reset())

        # Increment mu_0
        mu_0[state] += 1

    # Normalise mu_0
    mu_0 = mu_0 / mu_0.sum()

    return mu_0

# mu_0 = estimationOfMu0()
# print(mu_0)


# Below are the functions for computing Jn and J_pi
def computeJn(mu_0, Vn):
    """Compute Jn according to mu_0 and Jn."""

    Jn = np.dot(mu_0.T, Vn)[0, 0]

    return Jn


def computeJpi(mu_0, V_pi=v_q4):
    """Compute Jn according to mu_0 and Jn."""

    # Reshape V_q4
    V_pi = np.array(V_pi).reshape((-1, 1))

    J_pi = np.dot(mu_0.T, V_pi)[0, 0]

    return J_pi


# Below is the function for displaying the gap between Jn and J_pi
def displayJGap(nb_n=200):
    """Display the gap between Jn and J_pi according to n."""

    # Array of the iteration
    n_l = np.linspace(1, 5000, num=nb_n, dtype=int)

    # Computation of the V_n
    V_n_l = [estimatorMCValueFunctions(pol, n=n) for n in n_l]

    # Computation of mu_0
    mu_0 = estimationOfMu0()

    # Computation of J_n and J_pi
    J_n_l = [computeJn(mu_0, V_n_l[i]) for i in range(nb_n)]
    J_pi = computeJpi(mu_0)

    # Computation of the gap between them
    gaps = [J_n_l[i] - J_pi for i in range(nb_n)]

    # Parameters of the figure
    plt.figure(figsize=(8, 8))
    plt.grid(True)

    # Plot
    plt.scatter(n_l, gaps, label="J_n - J_pi", marker="x")

    # Legend
    plt.xlabel("n")
    plt.ylabel("|J_n - J_pi|")

    # Save the plot
    plt.savefig("./Images/Gaps_J_n_J_pi", bbox_inches='tight', pad_inches=0.0)

    # Display
    plt.show()


# displayJGap()


# Below is programmed the Q-learning algorithm
def greedyExplorationPolicy(x, Q, epsilon):
    """Return an action for the state x with the current approximation Q according
       to an epsilon greedy policy."""

    # Take a sample for deciding if we return a random action or not
    sample = np.random.rand()

    # Random action
    if sample < epsilon:
        action = np.random.choice(env.state_actions[x])

    # Take the maximising action
    else:
        # Possible actions for the given state
        possible_actions = env.state_actions[x]

        # Compute the maximising action among the possible states
        maximum = Q[x, possible_actions[0]]
        action = possible_actions[0]
        for i in range(1, len(possible_actions)):
                
                value = Q[x, possible_actions[1]]

                if value > maximum:
                        maximum = value
                        action = possible_actions[i]

    return action


def qLearning(n=1000, epsilon=0.2, gamma=0.95, delta=0.001):
    """Compute the coresponding Q matrix online and the opitmal policy."""

    # Definition of Tmax
    Tmax = -np.log(delta / 1) / (1 - gamma)

    # Parameters
    nb_actions = len(env.action_names)

    # Initialisation of the matrices of Q and alpha
    Q = np.zeros((env.n_states, nb_actions))
    alpha = np.zeros((env.n_states, nb_actions)) + 1

    # Rewards cumulated
    rewards = 0

    # Play the different trajectories
    for e in range(n):

        # Initialisation of the first state, the time and the terminal flag
        x_t = int(env.reset())
        t = 1
        term_flag = False

        # Initialisation of the reward
        reward = 0

        while (t < Tmax and not(term_flag)):

            # Select the action to do according to the policy
            a_t = greedyExplorationPolicy(x_t, Q, epsilon)

            # Observation of the next state
            x_t_plus_1, r_t, term_flag = env.step(x_t, a_t)

            # Temporal differences
            delta_t = r_t + gamma * Q[x_t_plus_1, :].max()

            # Extract alpha
            alpha_t = alpha[x_t, a_t]

            # Update of Q
            Q[x_t, a_t] = (1 - alpha_t) * \
                Q[x_t, a_t] + alpha_t * delta_t

            # Update of alpha
            alpha[x_t, a_t] = 1 / ((1 / alpha[x_t, a_t]) + 1)

            # Update of the state
            x_t = x_t_plus_1

            # Update of the reward
            reward += gamma ** (t - 1) * r_t

            # Increase the timer
            t += 1

        # Cumulated rerward over all the episodes
        rewards += reward

    return Q, rewards


################################################################################
# Work to do: Q5
################################################################################
v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]

# Here we extract the greedy policy according to Q
def computeGreedyPolicy(Q):
        """Return the greedy policy according to Q."""

        # Initialisation of the policy
        policy = np.zeros((env.n_states, 1))

        # Extract the best action according to Q
        for state in range(env.n_states):

                # Upate the policy value
                policy[state, 0] = Q[state, :].reshape(-1).argmax()

        return policy

# Here we define the inf norm
def infNorm(V):
    """Compute the inf norm on V."""
    
    return np.max(np.abs(V))

# Here we compute the value function according to the policy and Q
def valueFunctions(Q, policy):
        """Return the value functions for the given policy and Q matrix."""

        # Initialise the value function
        V_pi = np.zeros((env.n_states, 1))

        # Update the true value of the value function for each state
        for state in range(env.n_states):

                V_pi[state, 0] = Q[state, policy[state, 0]]

        return V_pi


# Here we define the function for plotting the difference between v_opt and v_pi_n
def displayVGaps(nb_n=200, epsilon=0.2):
        """Display the gap between v_opt and v_pi_n according to n."""

        # Array of the iteration
        n_l = np.linspace(1, 5000, num=nb_n, dtype=int)

        # Computation of the Q_n
        Q_n_l = [qLearning(n=n, epsilon=epsilon)[0] for n in n_l]

        # Computation of the policies given Q
        policy_l = [computeGreedyPolicy(Q_n_l[i]) for i in range(nb_n)]

        # Computation of v_n
        V_n_l = [valueFunctions(Q_n_l[i], policy_l[i]) for i in range(nb_n)]

        # Computation of the gap between them
        gaps = [infNorm(V_n_l[i] - v_opt) for i in range(nb_n)]

        # Parameters of the figure
        plt.figure(figsize=(8, 8))
        plt.grid(True)

        # Plot
        plt.scatter(n_l, gaps, label="||V_n - v*||", marker="x")

        # Legend
        plt.xlabel("n")
        plt.ylabel("||V_n - v*||")

        # Save the plot
        plt.savefig("./Images/Gaps_||V_n - v*||", bbox_inches='tight', pad_inches=0.0)

        # Display
        plt.show()

displayVGaps()


# Here we define the function for plotting the cumulated reward according to n
def displayCumulatedRewards(nb_n=200, epsilon=0.2):
        """Display the cumulated reward according to n."""

        # Array of the iteration
        n_l = np.linspace(1, 5000, num=nb_n, dtype=int)

        # Computation of the Q_n
        reward_l = [qLearning(n=n, epsilon=epsilon)[1] for n in n_l]

        # Parameters of the figure
        plt.figure(figsize=(8, 8))
        plt.grid(True)

        # Plot
        plt.scatter(n_l, reward_l, label="Cumulated Reward", marker="x")

        # Legend
        plt.xlabel("n")
        plt.ylabel("Cumulated Reward")

        # Save the plot
        plt.savefig("./Images/Cumulated Reward", bbox_inches='tight', pad_inches=0.0)

        # Display
        plt.show()

# displayCumulatedRewards()