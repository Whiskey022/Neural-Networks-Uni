/**
 * 
 */

import java.util.ArrayList;
import java.util.Random;


/**
 * Class for a neuron with linear activation
 * @author shsmchlr
 *
 */
public class LinearNeuron {
	/**
	 * data are arraylists of weights and the change in weights
	 * and the output and delta values
	 */
	protected ArrayList<Double> weights;
	protected ArrayList<Double> changeInWeights;
	protected double output, delta;
	protected DataSet trainData;
	
	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param data		Data set used to train neuron
	 */
	public LinearNeuron(int numIns, DataSet data) {
		weights = new ArrayList<Double>();			// create array list for weights
		changeInWeights = new ArrayList<Double>();	// and for the change in weights
		for (int ct=0; ct<=numIns; ct++) {			// need numIns+1 weights (inc that for bias)
			weights.add(0.0);						// add next weight as random number -1..1
			changeInWeights.add(0.0);				// add next change in weight, value 0
		}
		trainData = data;							// remember data set to be used
	}
	/**
	 * calcOutput of neuron
	 * @param nInputs	arraylist with the neuron inputs
	 * Does weighted sum being weight(0) + inputs(0..n) * weights(1..n+1)
	 */
	protected void calcOutput(ArrayList<Double> nInputs) {
		output = weights.get(0);							// start with bias weight(* 1)
		for (int ct=1; ct<weights.size(); ct++)				// for remain weights
			output += nInputs.get(ct-1) * weights.get(ct);	// add weight*appropriate input
	}
	
	/**
	 * depositOutput to the data set
	 * @param ct	which item in the data set
	 * @param d		the data set
	 */
	private void depositOutput (int ct, DataSet d) {
		d.storeOutput(ct, 0, output);			// just store output there
	}
	
	/**
	 * compute output of network passing it each item in data set in turn
	 * @param d
	 */
	private void computeNetwork(DataSet d) {
		for (int ct=0; ct < d.numInSet(); ct++) {	// for each item in data set
			calcOutput(d.getIns(ct));				// calculate output
			depositOutput(ct, d);					// and put in data set
		}
	}
	
	/**
	 * find delta
	 *	@param error	compute the delta from the error passed
	 */
	protected void findDelta(double error) {
		delta = error;
	}
	
	/**
	 * change all the weights in the neuron
	 * @param ins		array list of the inputs to the neuron
	 * @param learnRate	learning rate: change is learning rate * input * delta
	 * @param momentum	momentum constant : change is also momentun * change in weight last time
	 */
	private void changeTheWeights(ArrayList<Double> ins, double learnRate, double momentum) {
		double theIn;
		for (int wct = 0; wct < weights.size(); wct++) {			// for each weight
			if (wct == 0) theIn = 1.0; else theIn = ins.get(wct-1);	
					// input is 1 for bias weight, else it is wct'th-1 from ins 
			changeInWeights.set(wct, theIn * delta * learnRate + changeInWeights.get(wct) * momentum);
					// compute change in this weight
			weights.set(wct, weights.get(wct )+ changeInWeights.get(wct));
					// change the weight by that amount
		}
	}
	
	/**
	 * adapt the network, by inputting each item from the data set in turn, calculating
	 * the output, the error and delta, and adjusting all the weights
	 * @param d			data set
	 * @param learnRate	learning rate constant
	 * @param momentum	momentum constant
	 */
	private void adaptNetwork(DataSet d, double learnRate, double momentum) {
		for (int ct=0; ct < d.numInSet(); ct++) {				// for each item in set
			calcOutput(d.getIns(ct));							// calc output
			depositOutput(ct, d);								// put in data set
			findDelta(d.getErrors(ct).get(0));					// calc delta, from the error
			changeTheWeights(d.getIns(ct), learnRate, momentum);// change the weights
		}
	}
	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * @param wtsSplit
	 */
	private void setWeights (String[] wtsSplit) {
		for (int ct=0; ct<weights.size(); ct++) weights.set(ct, Double.parseDouble(wtsSplit[ct])); 
	}
	/**
	 * Load the weights with the values in the String wts
	 * @param wts
	 */
	public void setWeights (String wts) {
		setWeights(wts.split(" "));
	}
	/**
	 * Load the weights with random values in range -1 to 1
	 * @param rgen	random number generator
	 */
	public void setWeights (Random rgen) {
		for (int ct=0; ct<weights.size(); ct++) weights.set(ct,2.0*rgen.nextDouble() - 1);
	}
	/**
	 * return how many weights there are in the neuron
	 * @return
	 */
	public int numWeights() {
		return weights.size();
	}
	/**
	 * return the weights in the neuron as a string
	 * @return the string
	 */
	public String getWeights() {
		String s = "";
		for (int ct=0; ct<weights.size(); ct++) 
			s = s + String.format("%.5f", weights.get(ct)) + " ";
		return s;
	}
	/**
	 * initialise network before running
	 */
	public void doInitialise() {
		for (int ct=0; ct<changeInWeights.size(); ct++) changeInWeights.set(ct, 0.0);
								// set the change in weights to be 0
		output = 0.0;			// initialise others
		delta = 0.0;
		trainData.clearSSELog();// clear the log
	}
	/**
	 * present the data to the network and return string describing result
	 * @return
	 */
	public String doPresent() {
		computeNetwork(trainData);
		return trainData.toString(true, true) + "\nOver Set : " + trainData.dataAnalysis()+"\n";
	}
	/**
	 * get network to learn for numEpochs
	 * @param numEpochs		number of epochs to learn
	 * @param lRate			learning rate
	 * @param momentum		momentum
	 * @return				String with data about learning eg SSEs at relevant epochs
	 * 						At each epoch if numEpochs low, or do so at 10 of the epochs
	 */
	public String doLearn (int numEpochs, double lRate, double momentum) {
		String s = "";
		for (int ct=1; ct<=numEpochs; ct++) {
			adaptNetwork(trainData, lRate, momentum);
			if (numEpochs<20 || ct % (numEpochs/10) == 0)
				s = s + "Epoch " + Integer.toString(ct) + " : " + trainData.dataAnalysis()+"\n";
		}
		return s;
	}
	
	public static void main(String[] args) {
		// test network
		DataSet AndData = new DataSet("2 1 %.0f %.0f %.2f;x1 x2 AND;0 0 0;0 1 0;1 0 0;1 1 1");
		LinearNeuron LN = new LinearNeuron(2, AndData);
		LN.setWeights("0.2 0.3 -0.1");
		System.out.print(LN.doPresent());
		System.out.print(LN.doLearn(10, 0.2, 0.1));
		System.out.print(LN.doPresent());
		System.out.println("Weights " + LN.getWeights());
	}
}
