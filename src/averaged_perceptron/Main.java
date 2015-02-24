package averaged_perceptron;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;


public class Main 
{

	//static String PATH = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/data/train";
	static String PATH = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/practice_data/data/train";
	


	
	//the global list of all words across all articles
	static Set<String> GLOBO_DICT = new HashSet<String>();
	
	
	// hash map of all the words contained in individual files
	static Map<String, ArrayList<String> > training_fileDict = new HashMap<String, ArrayList<String>>();
	
	static Map<String, ArrayList<String> > test_fileDict = new HashMap<String, ArrayList<String>>();
	
	//input to perceptron. final struc.
	static Map<String, int[] > training_perceptron_input = new HashMap<String, int[] >();
	
	static Map<String, int[] > test_perceptron_input = new HashMap<String, int[] >();
	
	
	public static void main(String[] args) throws IOException 
	{
		//each of the diferent categories
		//String[] categories = { "/atheism", "/politics", "/science", "/sports"};
		String[] categories = { "/atheism", "/sports"};
		
		//cycle through all categories once to populate the global dict
		//for(int cycle = 0; cycle <= 3; cycle++)
		for(int cycle = 0; cycle <= 1; cycle++)
		{
			String general_data_partition = PATH + categories[cycle];
			
			File directory = new File( general_data_partition );
			IterateGlobo.iterateGlobo( directory , GLOBO_DICT  );
		}
		
		
		//cycle through again to populate the file dicts
		//for(int cycle = 0; cycle <= 3; cycle++)
		for(int cycle = 0; cycle <= 1; cycle++)
		{
			String general_data_partition = PATH + categories[cycle];
					
			File directory = new File( general_data_partition );
			IterateDir.iterateDirectory( directory , training_fileDict, GLOBO_DICT  );
					
		}
		
		//get feature vectors for the training data
		PrcptrnStrucGen.perceptron_data_struc_generateur( GLOBO_DICT, training_fileDict, training_perceptron_input );


		
		
		
		//cycle through all categories once to populate the global dict
		//for(int cycle = 0; cycle <= 3; cycle++)
		for(int cycle = 0; cycle <= 1; cycle++)
		{
			//get the test data
			//String test_path = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/data/train";
			String test_path = "/home/matthias/Workbench/SUTD/ISTD_50.570/assignments/practice_data/data/test";
			
			File test_dict = new File( test_path + categories[cycle]);
			IterateDir.iterateDirectory( test_dict , test_fileDict, GLOBO_DICT  );
		}
		
		//get feature vectors for the TEST data
		PrcptrnStrucGen.perceptron_data_struc_generateur( GLOBO_DICT, test_fileDict, test_perceptron_input );

		
		
		
		
//		System.out.println("TEST");
//		for (Map.Entry<String, int[]> entry : test_perceptron_input.entrySet()) 
//		{
//			System.out.println(entry.getKey() + ", " + Arrays.toString(entry.getValue()));
//		}
//		
//		System.out.println("TRAIN");
//		for (Map.Entry<String, int[]> entry : training_perceptron_input.entrySet()) 
//		{
//			System.out.println(entry.getKey() + ", " + Arrays.toString(entry.getValue()));
//		}
		



		
		
		
		///AHHAHAHAHH!!!!!!
		Perceptron.perceptron( GLOBO_DICT, training_perceptron_input, test_perceptron_input );
			
	}
	
}
