package averaged_perceptron;

import java.io.IOException;
import java.util.Set;

public class GloboPop
{

	public static void populate_globo_dict( String[] words, Set<String> GLOBO_DICT ) throws IOException
	{
		String word;
		
		for (int i = 0; i < words.length; i++) 
		{
			word = words[i];
			if (!GLOBO_DICT.contains(word))
			{
				GLOBO_DICT.add(word);
			}
		}	
	}
	
}
