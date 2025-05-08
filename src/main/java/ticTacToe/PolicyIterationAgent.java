package ticTacToe;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * 
 * @author ae187
 */
public class PolicyIterationAgent extends Agent {

    /**
     * This map is used to store the values of states according to the current policy (policy evaluation). 
     */
    HashMap<Game, Double> policyValues = new HashMap<Game, Double>();

    /**
     * This stores the current policy as a map from {@link Game}s to {@link Move}. 
     */
    HashMap<Game, Move> curPolicy = new HashMap<Game, Move>();

    double discount = 0.9;

    /**
     * The mdp model used, see {@link TTTMDP}
     */
    TTTMDP mdp;

    /**
     * Initializes the Policy Iteration Agent by setting up the MDP, initializing values, 
     * generating a random policy, and training until convergence.
     */
    public PolicyIterationAgent() {
        super();
        this.mdp = new TTTMDP();
        initValues(); // Initialize all state values to 0
        initRandomPolicy(); // Assign a random valid move for each state
        train(); // Perform policy iteration until convergence
    }

    /**
     * Initializes the agent with an existing policy.
     * @param p Existing policy
     */
    public PolicyIterationAgent(Policy p) {
        super(p);
    }

    /**
     * Initializes a learning agent with default MDP parameters.
     * @param discountFactor Discount factor for rewards
     */
    public PolicyIterationAgent(double discountFactor) {
        this.discount = discountFactor;
        this.mdp = new TTTMDP();
        initValues();
        initRandomPolicy();
        train();
    }

    /**
     * Initializes the agent with custom MDP parameters.
     * @param discountFactor Discount factor
     * @param winningReward Reward for winning
     * @param losingReward Reward for losing
     * @param livingReward Reward for each non-terminal state
     * @param drawReward Reward for drawing the game
     */
    public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward) {
        this.discount = discountFactor;
        this.mdp = new TTTMDP(winningReward, losingReward, livingReward, drawReward);
        initValues();
        initRandomPolicy();
        train();
    }

    /**
     * Initializes the policyValues map with all valid games and assigns an initial value of 0 to all states.
     */
    public void initValues() {
        // Generate all valid game states where it's X's turn or terminal
        List<Game> allGames = Game.generateAllValidGames('X');
        for (Game g : allGames) {
            this.policyValues.put(g, 0.0); // Set the initial value of each state to 0
        }
    }

    /**
     * Initializes a random policy by assigning a random valid move for every non-terminal state.
     */
    public void initRandomPolicy() {
        List<Game> allGames = Game.generateAllValidGames('X'); // Generate all valid states
        Random randomGenerator = new Random(); // Initialize a random generator

        // Iterate over all game states and assign a random move
        for (Game game : allGames) {
            List<Move> possibleMoves = game.getPossibleMoves(); // Get valid moves for the state
            if (!possibleMoves.isEmpty()) {
                int randomIndex = randomGenerator.nextInt(possibleMoves.size()); // Choose a random move
                Move randomMove = possibleMoves.get(randomIndex);
                curPolicy.put(game, randomMove); // Assign the random move to the state
            }
        }
    }

    /**
     * Evaluates the current policy by updating state values until convergence.
     * @param delta The convergence threshold for state value updates
     */
    protected void evaluatePolicy(double delta) {
        boolean hasConverged; // Tracks whether all states have converged

        do {
            hasConverged = true; // Assume convergence initially

            // Iterate over all states to update their values
            for (Game state : policyValues.keySet()) {
                double oldValue = policyValues.get(state); // Get the current value of the state
                double updatedValue = 0.0; // Initialize the updated value

                // Get the move defined by the current policy
                Move policyMove = curPolicy.get(state);

                if (policyMove != null) { // If the policy specifies a valid move
                    List<TransitionProb> transitions = mdp.generateTransitions(state, policyMove);

                    for (TransitionProb transition : transitions) {
                        double probability = transition.prob;
                        double reward = transition.outcome.localReward;
                        Game nextState = transition.outcome.sPrime;
                        double nextStateValue = policyValues.getOrDefault(nextState, 0.0);

                        // Update the state's value using the Bellman equation
                        updatedValue += probability * (reward + discount * nextStateValue);
                    }
                }

                policyValues.put(state, updatedValue); // Update the value of the state

                // Check if the value change exceeds the convergence threshold
                if (Math.abs(oldValue - updatedValue) > delta) {
                    hasConverged = false; // Continue iterations if not converged
                }
            }
        } while (!hasConverged); // Repeat until all state values converge
    }

    /**
     * Improves the current policy by selecting the best move for each state based on updated values.
     * @return True if the policy improved, false if no changes were made
     */
    protected boolean improvePolicy() {
        boolean hasPolicyChanged = false; // Tracks whether any policy updates occur

        // Iterate over all states in the current policy
        for (Game state : curPolicy.keySet()) {
            Move optimalMove = null; // Stores the best move for the state
            double highestValue = Double.NEGATIVE_INFINITY; // Initialize with the lowest possible value

            // Evaluate all possible moves for the state
            for (Move move : state.getPossibleMoves()) {
                double expectedValue = 0.0; // Expected value for the move

                List<TransitionProb> transitions = mdp.generateTransitions(state, move);

                for (TransitionProb transition : transitions) {
                    double probability = transition.prob;
                    double reward = transition.outcome.localReward;
                    Game nextState = transition.outcome.sPrime;
                    double nextStateValue = policyValues.getOrDefault(nextState, 0.0);

                    // Update the expected value for this move
                    expectedValue += probability * (reward + discount * nextStateValue);
                }

                // If this move yields a higher value, update the optimal move
                if (expectedValue > highestValue) {
                    highestValue = expectedValue;
                    optimalMove = move;
                }
            }

            // Update the policy if the best move differs from the current one
            if (optimalMove != null && !optimalMove.equals(curPolicy.get(state))) {
                curPolicy.put(state, optimalMove); // Update the move in the policy
                hasPolicyChanged = true; // Flag that the policy has improved
            }
        }

        return hasPolicyChanged; // Return whether the policy improved
    }
    
    /**
	 * The (convergence) delta
	 */
	double delta=0.1;

    /**
     * Alternates between policy evaluation and policy improvement until convergence.
     * Converts the final policy into a Policy object for use by the agent.
     */
    public void train() {
        while (true) {
            evaluatePolicy(delta); // Step 1: Evaluate the policy until values converge
            boolean policyChanged = improvePolicy(); // Step 2: Try to improve the policy
            if (!policyChanged) { // If no changes, the policy is optimal
                break;
            }
        }

        // Convert the current policy map into a Policy object
        this.policy = new Policy();
        for (Game state : curPolicy.keySet()) {
            Move assignedMove = curPolicy.get(state);
            this.policy.policy.put(state, assignedMove); // Store the move in the Policy object
        }
    }

    public static void main(String[] args) throws IllegalMoveException {
        // Test code to run the Policy Iteration Agent against a Human Agent
        PolicyIterationAgent pi = new PolicyIterationAgent();
        HumanAgent h = new HumanAgent();
        Game g = new Game(pi, h, h);
        g.playOut(); // Play out the game
    }
}
