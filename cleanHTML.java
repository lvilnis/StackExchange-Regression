import java.io.*;
import java.util.*;

public class cleanHTML {
	public static void main(String [] args) throws IOException{
		if (args.length < 1){
			CheckInputArguments();
			System.exit(1);
		}
		String CurrentDir = new java.io.File( "." ).getCanonicalPath();
		String InputFile = CurrentDir + "/src/" + args[0];
		String OutputFile = CurrentDir + "/src/" + args[1];
		PrintWriter ow = new PrintWriter(new FileWriter(OutputFile));
		String content = new Scanner(new File(InputFile)).useDelimiter("\\Z").next();
		content = content.replaceAll("<code>([^<]*)</code>", "");
		content = content.replaceAll("\\<.*?\\>", "");
		ow.println(content);
		ow.flush();
		ow.close();
		System.out.println("Successful write!");
	}
	public static void CheckInputArguments()
	{
		System.err.println("Name of Input and/or Output file missing");
	}
}
