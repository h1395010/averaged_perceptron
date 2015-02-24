package averaged_perceptron;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Set;

public class IterateGlobo 
{
	public static void iterateGlobo( File directory, Set<String> GLOBO_DICT) throws IOException 
	{
	    for (File file : directory.listFiles()) 
	    {
 
	    		String line; 
	    		BufferedReader br = new BufferedReader(new FileReader( file ));
	    		
	    		while((line = br.readLine()) != null) 
	    		{
	    			String[] words = line.split(" ");//those are your words


	    			GloboPop.populate_globo_dict( words, GLOBO_DICT );	
	    		}
	        
	    }  
	}
}
