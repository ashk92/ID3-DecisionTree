import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * An ID3 classifier for binary classification. Prints the tree learned 
 * through the training set to the console, prints the predicted and 
 * actual class and the number of correctly classified instances and 
 * the total number of instances in the test set. 
 * 
 * @author Ashwin Karthi
 *
 */
public class DT_ID3 {
	
	//m refers to the minimum number of nodes that should be present
	//to split at a node. Default value is 4
	static int m = 4;
	
	/**
	 * Finds whether the subset of instances have the same class 
	 * value or not.
	 * 
	 * @param subset
	 * @return 0 if all elements belong to class 0, 
	 * 		   1 if all elements belong to class 1
	 *        -1 if elements belong to either class.
	 */
	
	private static int hasSameClassValue(Instances subset){ 	
		double pval = 0;
		boolean differentClassValues = false;
		Instance inst = subset.instance(0);
		pval = subset.instance(0).classValue();
		for(int i=1;i<subset.numInstances(); i++){
			inst = subset.instance(i);
			if(inst.classValue()!=pval){
				differentClassValues = true;
				break;
			}
		}
		if(!differentClassValues)
			return (int) subset.instance(0).classValue();
		else
			return -1;
	}
	
	/**
	 * Builds an ID3 decision tree.
	 * 
	 * @param data
	 * @param root
	 * @param depth
	 */
	
	public static void buildTree(Instances data,DTreeNode root,int depth){
		
		if(data.numInstances() == 0){
			root.leaf = true;
			root.classvalue = 0;
			System.out.print(": "+data.classAttribute().value(0));
			return;
		}
		else if(data.numInstances() < m){
			//form the appropriate leaf and return;
			root.leaf = true;
			int peg = getNumPositveInstances(data);
			int neg = data.numInstances() - peg;
			if(peg>neg)
				root.classvalue = 1;
			else
				root.classvalue = 0;
			
			Attribute att = data.classAttribute();
			
			System.out.print(": "+att.value((int)root.classvalue));
			
			return;
		}
		
		else if(hasSameClassValue(data)!=-1){
			//form the appropriate leaf and return;
			root.leaf = true;
			root.classvalue = hasSameClassValue(data);
			
			Attribute att = data.classAttribute();
			
			System.out.print(": "+att.value((int)root.classvalue));
			
			return;
		}
		
		else{
			//determine candidate split
			
			//calculate node entropy
			int peg = 0, neg = 0;
			double entropy;
			for(int j=0;j<data.numInstances();j++){
				
				Instance inst = data.instance(j);
				if(inst.classValue() == 0)
					neg++;
				else
					peg++;
			}
			entropy = calculateEntropy(peg, neg, peg+neg);
			
			
			//traverse through each attribute
			double informationgain = - 200;
			int attributeindex = -1;
			double thresholdvals[] = null;			//holds the threshold value for the attributes
			int thresholdpts[] = null;				//holds the points at which the numeric threshold is selected
			double attvalentropy[] = null;			//for each child formed from the value the attribute can take, the child s entropy is calculated and stored
			int thresholdcount = 0;					//holds the number of attributed values/splits
			int thresholdpoint = 0;
			double thresholdval = 0;
			
			for(int i=0;i<data.classIndex();i++){
				//for each attribute determine the entropy

				Attribute attribute = data.attribute(i);
				int tcount[][] = null; //tcount[i][0] = total no of instances with given att val, 
										//tcount[i][1] = no of +ive values for that instance
				
				//if attribute is nominal
				if(attribute.isNominal()){								
					//for each nominal value of attribute calculate entropy
					tcount = new int[attribute.numValues()][2];				
					
					for(int k=0; k<data.numInstances();k++){
						//for each attribute count the positive examples
						
						Instance inst = data.instance(k); 			 
						tcount[(int)(inst.value(attribute))][0]++;
						
						if(inst.classValue()==1)
							tcount[(int)(inst.value(attribute))][1]++;
					}
					
					attvalentropy = new double[attribute.numValues()]; 
					
					for(int k=0; k<tcount.length;k++){					 
						attvalentropy[k] = calculateEntropy(tcount[k][1],tcount[k][0] - tcount[k][1],tcount[k][0]);
					}
					
					double tinfogain = getInformationGain(entropy,attvalentropy,tcount,data.numInstances());
					if(tinfogain > informationgain){
						informationgain = tinfogain;
						attributeindex = i;
					}
					
				}														
				
				//if attribute is numeric
				else{
					data.sort(i);						//sort the data set according to attribute i
					thresholdcount = 0;
					
					Instance inst = data.instance(0);
					double pval = inst.value(i);
					
					double distinctval[] = null;
					int flag[] = null;
					int distcnt = 1;
					
					for(int j=1;j<data.numInstances();j++){
						inst = data.instance(j);
						if(inst.value(i)!=pval){	//i points to the attribute
							distcnt++;
							pval = inst.value(i);
						}
					}
					//distcnt gives the count of distinct values
					
					distinctval = new double[distcnt];
					flag = new int[distcnt];
					double cval = 0; 
					int l = 0;
					
					for(int j=0; j<distcnt ;j++)
						flag[j] = 0;
					//boolean cvalchanged = false;
					//holds the class value
					//traverse through the instances and fill the flag array
					
					for(int j=0; j<data.numInstances(); j++){
						inst = data.instance(j);
						if(j==0){
							pval = inst.value(i);
							distinctval[l] = pval;
							continue;
						}
						else if(pval!=inst.value(i)){
							distinctval[++l] = inst.value(i);
							pval = inst.value(i);
						}
					}
					
					//distinctval array contains the distinct values of attributes

					//form the array of flags
					cval = data.instance(0).classValue();
					l = 0;
					
					for(int k = 0; k < data.numInstances();k++){
						inst = data.instance(k);
						
						if(inst.value(i) == distinctval[l]){
							if(cval!=inst.classValue())
								flag[l] = -1;
							if(flag[l]!=-1)
								flag[l] = (int)inst.classValue();
							else//flag already -1
								;
						}
						
						else{
							l++;
							k--;
						}
					}
					
					//calculating the threshold counts
					for(int j=1; j<flag.length ;j++){
						if(flag[j]!=flag[j-1])
							thresholdcount++;
						else{
							if(flag[j]==-1 || flag[j-1]==-1)
								thresholdcount++;
						}
					}
					
					thresholdvals = new double[thresholdcount];
					int k = 0;
					
					//calculating the threshold values
					for(int j=1; j<flag.length;j++){
						if(flag[j]!=flag[j-1]){
							thresholdvals[k++] = (distinctval[j-1]+distinctval[j])/2;
						}
						else{
							if(flag[j]==-1 || flag[j-1]==-1)
								thresholdvals[k++] = (distinctval[j-1]+distinctval[j])/2;
						}
					}
					
					k=0;
					
					//calculating the count of postive eg and total egs
					tcount = new int[thresholdcount+1][2];
					
					for(int j=0; j<data.numInstances();j++){
						inst = data.instance(j);
						
						if( k==thresholdcount || inst.value(i)<thresholdvals[k]){ //for last set, the values will be greater than the threshold point 
							tcount[k][0]++;
							if(inst.classValue() == 1)
								tcount[k][1]++;
						}
						else{
							k++;
							tcount[k][0]++;
							if(inst.classValue() == 1)
								tcount[k][1]++;
						}
					}
					
					//calculate entropy and information gain
					double tinfogain = -200, tempthresholdval = 0;
					
					for(k=0;k<thresholdcount;k++){
						//for each threshold point calculate the information gain from tcount array
						
						int p[][] = new int[2][2];
						for(int j=0; j<thresholdcount+1; j++){
							if(j<=k){
								p[0][0] += tcount[j][0];
								p[0][1] += tcount[j][1];
							}
							else{
								p[1][0] += tcount[j][0];
								p[1][1] += tcount[j][1];
							}
						}
						//p[0] contains the left side of numeric val and p[1] contains the rightside of the value
						//calculate the entropy
						double numericEntropy[] = new double[2];
						numericEntropy[0] = calculateEntropy(p[0][1], p[0][0] - p[0][1], p[0][0] );
						numericEntropy[1] = calculateEntropy(p[1][1], p[1][0] - p[1][1], p[1][0] );
						
						double tinfogain2 = getInformationGain(entropy, numericEntropy, p, data.numInstances());
						if(tinfogain2 > tinfogain){
							tinfogain = tinfogain2;
							tempthresholdval = thresholdvals[k];
						}
					}
					
					if(tinfogain > informationgain){
						informationgain = tinfogain;				
						thresholdval = tempthresholdval;
						attributeindex = i;
					}
					//System.out.println("hi, k value ="+k);
				}//end of else
			}//end of for loop
			//attribute index now points to attribute on which the tree should be split
			//information gain contains the information gain
			//threshold points now contain the points at which we should split

			root.attname = data.attribute(attributeindex).toString();
			root.peg = getNumPositveInstances(data);
			root.neg = data.numInstances() - peg;		
			root.entropy = entropy;
			root.gain = informationgain;
			root.leaf = false;
			root.attributeindex = attributeindex;

			if(data.attribute(attributeindex).isNumeric()){
				
				root.numeric = true;
				data.sort(attributeindex);
				
				Instance inst = data.instance(0);
				double pval = inst.classValue();
				
				root.node = new DTreeNode[2];
				root.node[0] = new DTreeNode();
				root.node[1] = new DTreeNode();
				
				Instances subset2 = new Instances(data,0);
				Instances subset3 = new Instances(data,0);
				
				for(int j=0;j<data.numInstances();j++){
					inst = data.instance(j);
					if(inst.value(attributeindex)<thresholdval)
						subset2.add(inst);
					else
						subset3.add(inst);
				}
				
				Attribute att = data.attribute(attributeindex);
				root.bound = thresholdval;
				
				root.node[0].parentattributevalue = att.value(attributeindex);
				root.node[0].parentattributename = data.attribute(attributeindex).name();
				root.node[0].left = true;
				
				System.out.println();
				for(int i=0;i<depth;i++)
					System.out.print("|\t");
				int temp = getNumPositveInstances(subset2);
				
				
				
				System.out.print(data.attribute(attributeindex).name()+" <= ");
				System.out.printf("%1$.6f",thresholdval);
				System.out.print(" ["+(subset2.numInstances() - temp)+" "+temp+"]");
					
				buildTree(subset2, root.node[0],depth+1);
				
				
				root.node[1].parentattributevalue = att.value(attributeindex);
				root.node[1].parentattributename = data.attribute(attributeindex).name();
				root.node[1].left = false;
				
				System.out.println();
				for(int i=0;i<depth;i++)
					System.out.print("|\t");
				temp = getNumPositveInstances(subset3);
				
				System.out.print(data.attribute(attributeindex).name()+" > ");
				System.out.printf("%1$.6f",thresholdval);
				System.out.print(" ["+(subset3.numInstances() - temp)+" "+temp+"]");
				
				
				buildTree(subset3, root.node[1],depth+1);
			}
			
			else{
				
				root.numeric = false;
				root.noOfThresholds = data.attribute(attributeindex).numValues();
				thresholdcount = data.attribute(attributeindex).numValues();
				root.node = new DTreeNode[thresholdcount];
				
				Instance inst;
				Attribute att = data.attribute(attributeindex);
				
				for(int i=0;i<thresholdcount;i++){
					root.node[i] = new DTreeNode();
				}
				
				Attribute attribute = data.attribute(attributeindex);
				
				for(int i=0;i<thresholdcount;i++){
					Instances subset2 = new Instances(data,0);
					for(int j=0;j<data.numInstances();j++){
						inst = data.instance(j);
						if(attribute.value(i).equals(inst.stringValue(attribute))){
							subset2.add(inst);
						}
					}
					root.node[i].parentattributevalue = attribute.value(i);
					root.node[i].parentattributename = attribute.name();
					
					System.out.println();
					for(int j=0;j<depth;j++)
						System.out.print("|\t");
					int temp = getNumPositveInstances(subset2);
					System.out.print(attribute.name()+" = "+attribute.value(i)+" ["+(subset2.numInstances()-temp)+" "+temp+"]");
					buildTree(subset2, root.node[i],depth+1);
				}
			}
		}
	}
	
	/**
	 * Calcuates the number of positive instances in the given subset.
	 * 
	 * @param subset
	 * @return the number of positive instances in the subset
	 */
	public static int getNumPositveInstances(Instances subset){
		int rval = 0;
		Instance inst = null;
		for(int i = 0;i<subset.numInstances();i++){
			inst = subset.instance(i);
			if(inst.classValue()==1)
				rval++;
		}
		return rval;
	}
	
	/**
	 * Calculates the information gain
	 * @param nodeEntropy
	 * @param attvalentropy
	 * @param tcount
	 * @param count
	 * @return
	 */
	public static double getInformationGain(double nodeEntropy, double attvalentropy[],int[][]tcount, int count){
		double infogain = nodeEntropy;
		
		for(int i=0;i<attvalentropy.length;i++){
			infogain -= ((double)(tcount[i][0])/(double)(count))*attvalentropy[i];
		}
		
		return infogain;
	}
	
	/**
	 * Main method of the program.
	 * @param args
	 */
	public static void main(String args[]){
		if(args.length!=2){
				System.out.println("Usage: java -jar <jar-file> <training-set.arff> <test-set.arff>");
		}
		try{
			BufferedReader reader = new BufferedReader(new FileReader(args[0])); //training-set
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			DTreeNode root = new DTreeNode();
			
			Instances subset = new Instances(data,0);
			Instance inst = data.instance(0);
			
			buildTree(data,root,0);
						
			reader = new BufferedReader(new FileReader(args[1])); //test set
			data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			int successcnt = 0;
			inst = data.instance(0);
			
			//printClass(inst, root, data);
			
			for(int i=0; i<data.numInstances(); i++){
				inst = data.instance(i);
				if(printClass(inst, root, data) == true)
					successcnt++;
			}
			
			System.out.print("\n"+successcnt+" "+data.numInstances());
		}
		catch(Exception e){
			System.out.println("Error: "+e);
		}
	}
	
	/**
	 * Prints the predicted and actual class value and returns true if the 
	 * prediction is correct or false if the prediction is wrong.
	 * 
	 * @param inst
	 * @param root
	 * @param data
	 * @return true if the prediction is correct and false if the prediction
	 * 	is wrong. 
	 */
	public static boolean printClass(Instance inst, DTreeNode root, Instances data){
		Attribute att = data.classAttribute();
		
		if(root.leaf == true){
			System.out.print("\n"+att.value((int)root.classvalue)+" "+att.value((int)(inst.classValue()))); //print predicted and actual class value
			if(root.classvalue == inst.classValue())
				return true; //if correctly classified
			else
				return false;
		}
		
		else{
			
			if(root.numeric == true){
				
				//compare value with threshold and recurse with the appropriate node
				if(inst.value(root.attributeindex)<=root.bound){	
					return printClass(inst, root.node[0] , data);
				}				
				else{
					return printClass(inst, root.node[1] , data);
				}
			}
			
			else{//root is nominal
				return printClass(inst,root.node[(int)(inst.value((root.attributeindex)))],data);
			}
		}
	}
	
	/**
	 * Calculates the entropy value given the number of positive instances, number
	 * of negative instances and the total number of instances
	 * 
	 * @param numPos
	 * @param numNeg
	 * @param total
	 * @return The entropy value calculated
	 */
	public static double calculateEntropy(int numPos,int numNeg,int total){
		double entropy = 0;
		
		if(numPos == 0 || numNeg ==0 || total ==0)
			return 0;
		
		entropy = - ((double)(numPos)/(double)(total))*(Math.log((double)(numPos)/(double)(total))/Math.log(2)) 
				- ((double)(numNeg)/(double)(total))*(Math.log((double)(numNeg)/(double)(total))/Math.log(2));
		
		return entropy;
	}
}
