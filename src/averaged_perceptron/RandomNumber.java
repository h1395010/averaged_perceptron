package averaged_perceptron;

import java.text.DecimalFormat;

public class RandomNumber 
{
	   public static double randomNumber(int min , int max) 
	   {
	       DecimalFormat df = new DecimalFormat("#.####");
	       double d = min + Math.random() * (max - min);
	       String s = df.format(d);
	       double x = Double.parseDouble(s);
	       return x;
	   }
}
