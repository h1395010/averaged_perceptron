package averaged_perceptron;

public class Prcptrn_CalcOutpt 
{
    /**
     * returns either 1 or 0 using a threshold function
     * theta is 0range
     * @param theta an integer value for the threshold
     * @param weights the array of weights
     * @param a the array of inputs
     * @return 1 or 0
     */
    static int calculateOutput(double theta, double[] weights, double[][] a, int index)
    {
        double sum = 0;
        int i;
        for (i = 0; i < a[i].length; i++) 
        {
            sum += weights[i] * a[index][i];
        }
        sum += weights[i];
        
        //unit step function
        //return (sum >= theta) ? 1 : 0;
        return (sum >= theta) ? 1 : -1;
    }
}
