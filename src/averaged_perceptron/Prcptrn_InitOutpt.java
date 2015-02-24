package averaged_perceptron;

import java.util.Map;
import java.util.Map.Entry;

public class Prcptrn_InitOutpt 
{
	static double[][] initializeOutput( Map<String, int[]> perceptronInput, 
		        int size, 
		        double[] outputs,
		        String LABEL)
	{
		final int inputSize = perceptronInput.size();
		final double[][] a = new double[size][inputSize];
		
		// 2d array for features
		int[][] feature_matrix = new int[inputSize][size];
		String[] output_label = new String[inputSize];
		int x = 0;
		for (Entry<String, int[]> entry : perceptronInput.entrySet()) 
		{
			int[] container = entry.getValue();
			
			for (int j = 0; j < container.length; j++) 
			{
				feature_matrix[x][j] = container[j];
				output_label[x] = String.valueOf(entry.getKey());
			}
			x++;
		}
		
		for (x = 0; x < inputSize; x++) 
		{
			for (int i = 0; i < a.length; i++) 
			{
				a[i][x] = feature_matrix[x][i];
			}
			//outputs[x] = output_label[x].equals(LABEL) ? 1 : 0;
			outputs[x] = output_label[x].equals(LABEL) ? 1 : -1;
		}
		
		return a;
	}
}
