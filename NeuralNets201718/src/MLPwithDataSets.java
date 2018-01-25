
/**
 * @author shsmchlr
 * Class of a multi layer perceptron network with training, unseen and validation data sets
 * MLP has hidden layer of sigmnoidally activated neurons and then output layer(s)
 * Such a network can learn using the training set, and be tested on the unseen set
 * In addition, it can use the validation set to decide when to stop learning
 */
public class MLPwithDataSets extends MultiLayerNetwork {
		// you may want to add variables here
	
	protected DataSet unseenData;			// unseen data set
	protected DataSet validationData;		// validation set : is set to null if that set is not being used
	private Double previousSSESum;			// previous validation SSEs sum from doLearn() function
	private boolean learnt;					// boolean if the problem is already learnt (validation SSE started to rise)
	
	/**
	 * Constructor for the MLP
	 * @param numIns			number of inputs	of hidden layer
	 * @param numOuts			number of outputs	of hidden layer
	 * @param data				training data set used
	 * @param nextL				next layer		
	 * @param unseen			unseen data set
	 * @param valid				validation data set
	 */
	MLPwithDataSets (int numIns, int numOuts, DataSet data, LinearLayerNetwork nextL,
						DataSet unseen, DataSet valid) {
		super(numIns, numOuts, data, nextL);	// create the MLP
												// and store the data sets
		unseenData = unseen;
		validationData = valid;
	}

	/** 
	 * initialise network before learning ...
	 */
	public void doInitialise() {
		super.doInitialise();
		previousSSESum = 1000.0;		//Set previousSSESum to very high value, so the initial SSESum would be lower than this
		learnt = false;					//Set learnt to false
	}
	/**
	 * present the data to the set and return a String describing results
	 * Here it returns the performance when the training, unseen (and if available) validation
	 * sets are passed - typically responding with SSE and if appropriate % correct classification
	 */
	public String doPresent() {
		String S;
		computeNetwork(trainData);
		S = "Train: " +  trainData.dataAnalysis();
		computeNetwork(unseenData);
		S = S + " Unseen: " + unseenData.dataAnalysis();
		if (validationData != null) {
			computeNetwork(validationData);
			S = S + " Valid: " + validationData.dataAnalysis();
		}
		return S;
	}

	/**
	 * learn training data, printing SSE at 10 of the epochs, evenly spaced
	 * if a validation set available, learning stops when SSE on validation set rises
	 * this check is done by summing SSE over 10 epochs
	 * @param numEpochs		number of epochs
	 * @param lRate			learning rate
	 * @param momentum		momentum
	 * @return				String with data about learning eg SSEs at relevant epoch
	 */
	public String doLearn (int numEpochs, double lRate, double momentum) {
		//If already learnt (previous SSE was lower than current) return empty string
		if (learnt) {
			return "";
		}
		
		//Otherwise will do the learning 
		
		// Epochs so far retrieved by checking SSELog size
		int epochsSoFar = trainData.sizeSSELog();
		// Initialise string which will contain: all Epochs, and SSEs, and if appropriate % correctly classified
		String s = "";
		//Sum for SSEs, set to 0 before the loop
		Double SSESum = 0.0;
		
		//Loop repeats the number of epochs
		for (int ct=1; ct<=numEpochs; ct++) {
			adaptNetwork(trainData, lRate, momentum);		//Adapt the network using the training data 
			computeNetwork(validationData);					//Pass the validation set to the network
			validationData.addToSSELog();					//Add the SSE of the validation data set to its SSE log 
			
			//Loop to add validation SSEs to sum
			for (int j=0; j<trainData.getSSE().size(); j++) {
				SSESum += validationData.getSSE().get(j);
			}
			//Every 10th epoch, check SSEs sum
			if (ct%10 == 0) {
				if (previousSSESum<SSESum) {	//If previous SSESum lower than the current SSESum, set learnt to true, 
					learnt = true;				// and return string saying stopped after how many epochs, this will stop the learning loop as well
					return s + "Stopped after " + trainData.sizeSSELog() + " epochs.\n";
				} else {		//Otherwise, set previous SSEs sum to current, and the current sum to 0
					previousSSESum = SSESum;
					SSESum = 0.0;
				}
			}
			// print appropriate number of times
			if (numEpochs<20 || ct % (numEpochs/10) == 0)
				s = s + addEpochString(ct+epochsSoFar) + " : " +
				trainData.dataAnalysis()+"\n";	// Epoch, and SSE, and if appropriate % correctly classified
		}
		//Return all Epochs, and SSEs, and if appropriate % correctly classified 
		return s;
	}

}
