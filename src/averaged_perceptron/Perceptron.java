package averaged_perceptron;

/**
The Perceptron Algorithm
By Dr Noureddin Sadawi
Please my youtube videos on perceptron for things to make sense!

Code adapted from:
https://github.com/RichardKnop/ansi-c-perceptron
*/  

import java.util.*;
import java.util.Map.Entry;

class Perceptron 
{
   static final int MAX_ITER = 1000;
   static final double LEARNING_RATE = 0.1;
   static final double THETA = 0;
   
   static final String LABEL = "atheism";
   //static final String LABEL = "sports";
   
   //1 or 0 corresponding to the feature vector for aetheism or sports
   //static final int TEST_CLASS = 0;
   static final int TEST_CLASS = 1;
   
   static int test_output;

   public static void perceptron(Set<String> globoDict,
       Map<String, int[]> trainingPerceptronInput,
       Map<String, int[]> testPerceptronInput)
   {
	   //store weights to be averaged. 
	   Map<Integer,double[]> cached_weights = new HashMap<Integer,double[]>();
	   
	   
       final int globoDictSize = globoDict.size(); // number of features

       // weights total 32 (31 for input variables and one for bias)
       double[] weights = new double[globoDictSize + 1];
       for (int i = 0; i < weights.length; i++) 
       {
           //weights[i] = Math.floor(Math.random() * 10000) / 10000;
           //weights[i] = randomNumber(0,1);
           weights[i] = 0.0;
       }
       

       int inputSize = trainingPerceptronInput.size();
       double[] outputs = new double[inputSize];
       final double[][] a = Prcptrn_InitOutpt.initializeOutput(trainingPerceptronInput, globoDictSize, outputs, LABEL);

       
       double globalError;
       int iteration = 0;
       do 
       {
           iteration++;
           globalError = 0;
           // loop through all instances (complete one epoch)
           for (int p = 0; p < inputSize; p++) 
           {
               // calculate predicted class
               double output = Prcptrn_CalcOutpt.calculateOutput(THETA, weights, a, p);
               // difference between predicted and actual class values
               //always either zero or one
               double localError = outputs[p] - output;
               
               int i;
               for (i = 0; i < a.length; i++) 
               {
                   weights[i] += LEARNING_RATE * localError * a[i][p];
               }
               weights[i] += LEARNING_RATE * localError;

               // summation of squared error (error value for all instances)
               globalError += localError * localError;
           }
           
           //store weights for averaging
           cached_weights.put( iteration , weights );
           
           /* Root Mean Squared Error */
           System.out.println("Iteration " + iteration + " : RMSE = " + Math.sqrt(globalError / inputSize));
       } 
       while (globalError != 0 && iteration <= MAX_ITER);
       
       
       
       
       int size = globoDictSize + 1;
       //compute averages
       double[] sums = new double[size];
       double[] averages = new double[size];

       for (Entry<Integer, double[]> entry : cached_weights.entrySet()) 
       {
           double[] value = entry.getValue();
           for(int pos=0; pos < size; pos++){
               sums[ pos ] +=  value[ pos ]; 
           }
       }
       for(int pos=0; pos < size; pos++){
           averages[ pos ] = sums[ pos ] / cached_weights.size(); 
       }
       
       
       
       
       
       
       

       System.out.println("\n=======\nDecision boundary equation:");
       int i;
       for (i = 0; i < a.length; i++) 
       {
           System.out.print(" a");
           if (i < 10) System.out.print(0);
           System.out.println( i + " * " + weights[i] + " + " );
           
       	
       }
       System.out.println(" bias: " + weights[i]);
       
       
       //TEST
       //this works because, at this point the weights have already been learned. 
       inputSize = testPerceptronInput.size();
       outputs = new double[inputSize];
       double[][] z = Prcptrn_InitOutpt.initializeOutput(testPerceptronInput, globoDictSize, outputs, LABEL); 

       test_output = Prcptrn_CalcOutpt.calculateOutput(THETA, weights, z, TEST_CLASS);       
       
       System.out.println("class = " + test_output);
       
       
   }

   


   



}



