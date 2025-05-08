# Tic-Tac-Toe MDP & Reinforcement Learning Agents

This project implements agents for the Tic-Tac-Toe game using **Markov Decision Processes (MDPs)** and **Reinforcement Learning (RL)** techniques. The project includes three primary agents: **Value Iteration**, **Policy Iteration**, and **Q-Learning**. Below is an overview of the core classes and the role they play in the implementation.

## Core Classes

### 1. **ValueIterationAgent.java**

This class implements the **Value Iteration** algorithm. The key methods to focus on are:

* **`iterate()`**: This method performs the value iteration for `k` steps. In each iteration, it updates the value for each state by considering all possible actions and computing the maximum expected utility.
* **`extractPolicy()`**: After value iteration, this method extracts the optimal policy by selecting the action that maximizes the value for each state.

The agent uses the `TTTMDP` class to generate possible transitions for a given state and computes the value function for each state.

### 2. **PolicyIterationAgent.java**

This class implements the **Policy Iteration** algorithm. The key methods are:

* **`initRandomPolicy()`**: Initializes a random policy for the agent.
* **`evaluatePolicy()`**: This method evaluates the current policy by calculating the value function for each state under the current policy.
* **`improvePolicy()`**: Performs the policy improvement step by updating the policy to select the action that maximizes the value of the state.
* **`train()`**: Coordinates the process of policy evaluation and improvement, iterating until the policy converges.

The agent works by iteratively improving its policy based on the value function computed from the current policy.

### 3. **QLearningAgent.java**

This class implements the **Q-Learning** algorithm. It is a **Reinforcement Learning** agent that learns the best policy through exploration and exploitation.

* **`train()`**: This method trains the agent by interacting with the environment over multiple episodes. It updates the Q-values using the Q-learning update rule.
* **`extractPolicy()`**: After training, this method extracts the optimal policy by selecting the action with the highest Q-value for each state.

Q-learning uses an epsilon-greedy policy during training (with exploration), and during testing, the agent follows the extracted policy (exploitation).

### 4. **Game.java**

The `Game` class manages the 3x3 Tic-Tac-Toe game logic. It allows different agents to play against each other by specifying the agents for player `X` and player `O`. It also prints the game state and keeps track of the game outcome.

Key features:

* The game can be played between two rule-based agents (e.g., `random`, `aggressive`) or one or more agents can be your own implementations (`vi`, `pi`, `ql`).
* The `-x` and `-o` options specify which agents play `X` and `O`, respectively.
* The `-s` option controls which agent plays first.

### 5. **TTTMDP.java**

Defines the **Markov Decision Process (MDP)** for Tic-Tac-Toe. This class provides methods to generate state transitions for a given action (move) and calculates the rewards and probabilities for each transition. The `generateTransitions()` method is key here, as it provides the state transitions and associated probabilities for the agent's decision-making process.

### 6. **TTTEnvironment.java**

Defines the **Reinforcement Learning** environment for Tic-Tac-Toe. It encapsulates the environment’s rules, such as the valid actions and game rules, and provides the agent with state transitions and rewards based on the actions it takes. The environment interacts with both the `Q-learning` agent and the `Value/Policy Iteration` agents.

### 7. **Agent.java**

This is the **abstract base class** for all agents. It provides a common interface for different types of agents, such as `ValueIterationAgent`, `PolicyIterationAgent`, and `QLearningAgent`. The base class defines general methods that are shared across all agents, such as the initialization and action selection methods.

### 8. **HumanAgent.java**

This class defines a **human-controlled agent**. It allows a user to play Tic-Tac-Toe via the command line interface by requesting input for the next move.

### 9. **RandomAgent.java**

This is a **random agent** that selects actions (moves) randomly from the available options. It serves as a baseline agent and is useful for testing the performance of the other agents.

### 10. **Move.java**

Defines a **Tic-Tac-Toe move**. It encapsulates the details of a move, such as the player and the position on the board.

### 11. **Outcome.java**

This class represents a **transition outcome**. It holds a tuple containing the current state, the action taken, the reward received, and the next state. The outcome is used to update Q-values in Q-learning and for policy evaluation in the other agents.

### 12. **Policy.java**

This is an **abstract class** that defines a general policy. Specific policies, such as the random policy used by the `RandomAgent`, subclass this class and implement their own decision-making logic.

### 13. **RandomPolicy.java**

Defines a **random policy** used by the `RandomAgent`. It selects actions randomly from the set of available moves.

## Key Concepts

* **MDP (Markov Decision Process)**: The Tic-Tac-Toe game is modeled as an MDP, where each state represents a configuration of the game board, and the agent chooses actions (moves) to maximize the expected reward.
* **Value Iteration**: This algorithm computes the value function for each state and derives the optimal policy by iterating over all states and actions to find the best move.
* **Policy Iteration**: This method iteratively improves a policy by evaluating and refining it until it converges to the optimal policy.
* **Q-Learning**: A reinforcement learning algorithm that uses Q-values to determine the optimal policy by interacting with the environment and learning from rewards received for actions.
