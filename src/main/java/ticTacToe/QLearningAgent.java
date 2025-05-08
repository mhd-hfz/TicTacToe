package ticTacToe;

import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e., a table of Q-Values. This table is implemented in the {@link QTable} class.
 */
public class QLearningAgent extends Agent {

    double alpha = 0.1;        // Learning rate
    int numEpisodes = 10000;  // Number of episodes for training
    double discount = 0.9;    // Discount factor (gamma)
    double epsilon = 0.1;     // Epsilon for epsilon-greedy policy
    QTable qTable = new QTable(); // Q-Table
    TTTEnvironment env = new TTTEnvironment(); // Environment

    /**
     * Constructor with parameters.
     */
    public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) {
        env = new TTTEnvironment(opponent);
        this.alpha = learningRate;
        this.numEpisodes = numEpisodes;
        this.discount = discount;
        initQTable();  // Initialize the Q-table
        train();       // Train the agent using Q-learning
    }

    /**
     * Default constructor.
     */
    public QLearningAgent() {
        this(new RandomAgent(), 0.1, 100000, 0.9);
    }

    /**
     * Initializes all valid Q-values to 0 for all states and actions in the environment.
     */
    protected void initQTable() {
        List<Game> allGames = Game.generateAllValidGames('X'); // All valid games where it's X's turn or terminal.
        for (Game g : allGames) {
            for (Move m : g.getPossibleMoves()) {
                this.qTable.addQValue(g, m, 0.0); // Initialize Q-values for each valid (state, action) pair to 0
            }
        }
    }

    /**
     * Trains the agent using Q-Learning, iterating through episodes and updating Q-values.
     */
    public void train() {
        // Iterate over all episodes
        for (int i = 0; i < numEpisodes; i++) {
            while (!this.env.isTerminal()) {  // Run until the environment reaches a terminal state
                Game g = this.env.getCurrentGameState();  // Get the current game state

                if (g.isTerminal()) continue; // Skip if the state is terminal (no further actions possible)

                // Implement epsilon-greedy move selection inline
                List<Move> moves = g.getPossibleMoves();  // Get all possible moves in the current state
                Random random = new Random();
                double chance = random.nextDouble();  // Random value to decide whether to explore or exploit
                Move m;

                if (chance < epsilon) {  // Exploration: choose a random move
                    m = moves.get(random.nextInt(moves.size()));
                } else {  // Exploitation: choose the best known move based on Q-values
                    double maxQ = -Double.MAX_VALUE;
                    Move bestMove = null;
                    for (Move move : moves) {
                        double qValue = qTable.getQValue(g, move);  // Get the Q-value for the current move
                        if (qValue > maxQ) {  // If this move has the highest Q-value, update the best move
                            maxQ = qValue;
                            bestMove = move;
                        }
                    }
                    m = bestMove;
                }

                Outcome outcome = null;
                try {
                    outcome = this.env.executeMove(m); // Execute the chosen move in the environment
                } catch (IllegalMoveException e) {
                    e.printStackTrace();  // Handle any illegal move exceptions
                }

                // Update the Q-value for the current state-action pair
                double qvalue = this.qTable.getQValue(outcome.s, outcome.move);  // Get the current Q-value

                // Calculate the maximum Q-value for the next state (max Q(s') for all possible moves)
                double maxQNextState = 0.0;
                if (!outcome.sPrime.isTerminal()) {  // If the next state is not terminal
                    maxQNextState = -Double.MAX_VALUE;
                    for (Move nextMove : outcome.sPrime.getPossibleMoves()) {
                        maxQNextState = Math.max(maxQNextState, this.qTable.getQValue(outcome.sPrime, nextMove));
                    }
                }

                // Update the Q-value based on the Q-learning formula
                double newqvalue = (1 - this.alpha) * qvalue + this.alpha * (outcome.localReward 
                                + this.discount * maxQNextState);
                this.qTable.addQValue(outcome.s, outcome.move, newqvalue); // Store the updated Q-value in the Q-table
            }
            this.env.reset(); // Reset environment for next episode
        }
        this.policy = extractPolicy(); // Extract the policy after training
        if (this.policy == null) {
            System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
        }
    }

    /**
     * Extracts a policy from the Q-table by selecting the move with the highest Q-value for each state.
     */
    public Policy extractPolicy() {
        Policy policy = new Policy();

        // Iterate over each game state in the Q-table
        for (Game state : this.qTable.keySet()) {
            // Skip terminal states
            if (state.isTerminal()) continue;

            // Initialize variables to track the best move
            double bestQValue = -Double.MAX_VALUE;
            Move optimalMove = null;

            // Evaluate each possible move for the current state
            for (Move move : state.getPossibleMoves()) {
                double currentQValue = qTable.getQValue(state, move);  // Get the Q-value for the current move

                // If the current move has a higher Q-value, choose it
                if (currentQValue > bestQValue) {
                    bestQValue = currentQValue;  // Update best Q-value
                    optimalMove = move;  // Update the optimal move for this state
                }
            }

            // Store the best move for the current state in the policy
            policy.policy.put(state, optimalMove);
        }

        return policy;
    }

    /**
     * Main method for testing against a human agent.
     */
    public static void main(String[] args) throws IllegalMoveException {
        QLearningAgent agent = new QLearningAgent();  // Create a new Q-learning agent
        HumanAgent human = new HumanAgent();  // Create a new human agent
        Game game = new Game(agent, human, human);  // Start a new game with both agents
        game.playOut();  // Play out the game
    }
}
