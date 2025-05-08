package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=50;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		// Value initialization: Set all state values to 0
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate() {
	    // Repeat the process for 'k' iterations
	    for (int i = 0; i < k; i++) {
	        Map<Game, Double> newValues = new HashMap<>();

	        // Iterate over all games in the current value function
	        for (Game game : valueFunction.keySet()) {
	            // If the game state is terminal, its value is always 0
	            if (mdp.isTerminal(game)) {
	                newValues.put(game, 0.0);
	                continue;
	            }

	            // Track the maximum expected value for this game state
	            double maxExpectedValue = Double.NEGATIVE_INFINITY;

	            // Get all possible moves for the current game state
	            List<Move> possibleMoves = game.getPossibleMoves();

	            for (Move move : possibleMoves) {
	                double expectedValue = 0.0;

	                // Generate transitions for the current move
	                List<TransitionProb> transitions = mdp.generateTransitions(game, move);

	                for (TransitionProb transition : transitions) {
	                    // Calculate the contribution of this transition
	                    Outcome outcome = transition.outcome;
	                    double reward = outcome.localReward;
	                    Game nextState = outcome.sPrime;
	                    double prob = transition.prob;

	                    // Fetch the value of the next state (default to 0 if not present)
	                    double nextStateValue = valueFunction.getOrDefault(nextState, 0.0);

	                    // Add this transition's expected contribution
	                    expectedValue += prob * (reward + discount * nextStateValue);
	                }

	                // Update the maximum expected value if this move is better
	                maxExpectedValue = Math.max(maxExpectedValue, expectedValue);
	            }

	            // Store the maximum value found for the current state
	            newValues.put(game, maxExpectedValue);
	        }

	        // Update the value function for the next iteration
	        valueFunction = newValues;
	    }
	}

	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy() {
	    // Create a new policy object
	    Policy policy = new Policy();

	    // Loop through all game states in the value function
	    for (Game game : valueFunction.keySet()) {
	        // Skip terminal states as they don't need a policy
	        if (mdp.isTerminal(game)) {
	            continue;
	        }

	        Move bestMove = null;  // To track the best move for this game state
	        double maxExpectedValue = Double.NEGATIVE_INFINITY; // Start with the lowest possible value

	        // Check all possible moves from the current game state
	        for (Move move : game.getPossibleMoves()) {
	            double expectedValue = 0.0;

	            // Get all transitions for this move
	            for (TransitionProb transition : mdp.generateTransitions(game, move)) {
	                Outcome outcome = transition.outcome;
	                double reward = outcome.localReward;
	                Game nextState = outcome.sPrime;
	                double prob = transition.prob;

	                // Get the value of the next state; if missing, assume 0
	                double nextStateValue = valueFunction.getOrDefault(nextState, 0.0);

	                // Calculate the contribution of this transition
	                expectedValue += prob * (reward + discount * nextStateValue);
	            }

	            // Update the best move if this one is better
	            if (expectedValue > maxExpectedValue) {
	                maxExpectedValue = expectedValue;
	                bestMove = move;
	            }
	        }

	        // Add the best move to the policy if one was found
	        if (bestMove != null) {
	            policy.policy.put(game, bestMove);
	        }
	    }

	    // Return the generated policy
	    return policy;
	}

	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
	
	}
}
