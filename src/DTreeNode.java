/**
 * Class that corresponds to a node of the decision tree.
 * 
 * @author Ashwin Karthi.
 *
 */

public class DTreeNode {
	
	int peg;			//positive examples
	int neg;			//negative examples
	int noOfThresholds;
	
	String attname;
	int attributeindex;
	
	String parentattributename;
	String parentattributevalue;
	
	double bound;
	double entropy;
	double gain;
	double classvalue;
	
	boolean numeric;
	boolean leaf;
	boolean left;
	
	public DTreeNode[] node;

}
