package averaged_perceptron;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class PrcptrnStrucGen 
{
	public static void perceptron_data_struc_generateur(Set<String> GLOBO_DICT, 
													    Map<String, ArrayList<String> > fileDict,
													    Map<String, int[] > perceptron_input)
	{
		//create a new entry in the array list 'perceptron_input'
		//with the key as the file name from fileDict
			//create a new array which is the length of GLOBO_DICT
			//iterate through the indicies of GLOBO_DICT
				//for all words in globo dict, if that word appears in fileDict,
				//increment the perceptron_input index that corresponds to that
				//word in GLOBO_DICT by the number of times that word appears in fileDict
		
		//so i can get the index later
		List<String> GLOBO_DICT_list = new ArrayList<String>(GLOBO_DICT);
		
		for (Map.Entry<String, ArrayList<String>> entry : fileDict.entrySet()) 
		{
		    int[] cross_czech = new int[GLOBO_DICT_list.size()];
		    //initialize to zero
		    Arrays.fill(cross_czech, 0);

		    for (String s : GLOBO_DICT_list)
		    {
		    	
		        for(String st : entry.getValue()) 
		        {
		        	if( st.equals(s) )
		        	{
		        		cross_czech[ GLOBO_DICT_list.indexOf( s ) ] = cross_czech[ GLOBO_DICT_list.indexOf( s ) ] +1;
		        	}
		        }
		    }
		    perceptron_input.put( entry.getKey() , cross_czech);	
		}
	}
}

