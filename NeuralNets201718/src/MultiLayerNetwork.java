
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


/**
 * @author shsmchlr
 * This a multi layer network, comprising a hidden layer of neurons with sigmoid activation
 * Followed by another layer with linear/sigmoid activation, or be another multi layer network
 * A layer is defined as a set of neurons which have the same inputs
 */
public class MultiLayerNetwork extends SigmoidLayerNetwork {
	LinearLayerNetwork nextLayer;			// this is the next layer of neurons
	
	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 * @param nextL		the next layer in the network
	 */
	public MultiLayerNetwork(int numIns, int numOuts, DataSet data, LinearLayerNetwork nextL) {
		super(numIns, numOuts, data);			// construct the current layer
		nextLayer = nextL;						// store link to next layer
	}
	/**
	 * calcOutputs of network
	 * @param nInputs	arraylist with the neuron inputs
	 * Calculates outputs of this layer and then of next layer
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		//Calculate hidden layer outputs using inherited calcOutputs function
		super.calcOutputs(nInputs);
		//Calculate following layer outputs with its calcOutputs function,
		//providing it previous layer outputs as inputs
		nextLayer.calcOutputs(super.getOutputs());
	}
	
	/**
	 * depositOutputs of the output layer of the network to the data set
	 * @param ct	which item in the data set
	 * @param d		the data set
	 */
	protected void depositOutputs (int ct, DataSet d) {
		//Call the following layer's function to deposit Layer Network's outputs
		nextLayer.depositOutputs(ct, d);
	}
	
	/**
	 * find the deltas in the whole network from the errors passed
	 * needs to find the deltas in the next layer, and then calculate them in this layer
	 *	@param errors in the output layer	
	 */
	protected void findDeltas(ArrayList<Double> errors) {
		//First find deltas of the following layer
		nextLayer.findDeltas(errors);
		//Then, find deltas in the current layer, providing following layer's weighted deltas as errors
		super.findDeltas(nextLayer.weightedDeltas());
	}
	
	/**
	 * change all the weights in the network, in this layer and the next
	 * @param ins		array list of the inputs to the neuron
	 * @param learnRate	learning rate: change is learning rate * input * delta
	 * @param momentum	momentum constant : change is also momentun * change in weight last time
	 */
	protected void changeTheWeights(ArrayList<Double> ins, double learnRate, double momentum) {
		//Change weights of the hidden layer
		super.changeTheWeights(ins, learnRate, momentum);
		//Change weights of the following layer
		nextLayer.changeTheWeights(getOutputs(), learnRate, momentum);
	}	
	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * @param wtsSplit
	 */
	protected void setWeights (String[] wtsSplit) {
		super.setWeights(wtsSplit);					// copy relevant weights in this layer
		nextLayer.setWeights(Arrays.copyOfRange(wtsSplit, weights.size(), wtsSplit.length));
				// copy remaining strings in wtsSplit and pass to next layer
	}
	/**
	 * Load the weights with random values
	 * @param rgen	random number generator
	 */
	public void setWeights (Random rgen) {
		super.setWeights(rgen);			// do so in this layer
		nextLayer.setWeights(rgen);		// and in next
	}
	/**
	 * return how many weights there are in the network
	 * @return
	 */
	public int numWeights() {
		//Return a sum of a hidden layer weights number plus the following layer weights number
		return super.numWeights() + nextLayer.numWeights();
	}
	/**
	 * return the weights in the whole network as a string
	 * @return the string
	 */
	public String getWeights() {
		//Return a string of the hidden layer weights plus the following layer weights
		return super.getWeights() + nextLayer.getWeights();
	}

	/**
	 * initialise network before running
	 */
	public void doInitialise() {
		super.doInitialise();					// initialise this layer 
		nextLayer.doInitialise();				// and then initialise next layer
	}
	
	/**
	 * function to test MLP on xor problem
	 */
	public static void TestXOR() {
		DataSet Xor = new DataSet("2 1 %.0f %.0f %.3f;x1 x2 XOR;0 0 0;0 1 1;1 0 1;1 1 0");
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Xor, new SigmoidLayerNetwork(2, 1, Xor));
		MLN.setWeights("0.862518 -0.155797 0.282885 0.834986 -0.505997 -0.864449 0.036498 -0.430437 0.481210");
		MLN.doInitialise();
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());
		System.out.println(MLN.doLearn(1000, 0.4,  0.7));
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());

	
	}
	/**
	 * function to test MLP on other non linear separable problem
	 */
	public static void TestOther() {
//		DataSet Other = new DataSet("2 2 %.1f %.0f %.3f;0.1 1.2 1 0;0.7 1.8 1 0;0.8 1.6 1 0;1 0.8 0 0;"+
//									 "0.3 0.5 1 1;0 0.2 1 1;-0.3 0.8 1 1;-0.5 -1.5 0 1;-1.5 -1.3 0 1");
		DataSet Other = new DataSet(DataSet.GetFile("other.txt"));
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Other, new SigmoidLayerNetwork(2, 2, Other));
			MLN.computeNetwork(Other);
			MLN.doInitialise();
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
			System.out.println(MLN.doLearn(1000,  0.3,  0.5));
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
		
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//TestXOR();				// test MLP on the XOR problem
		TestOther();			// test MLP on the other problem
	}

}
