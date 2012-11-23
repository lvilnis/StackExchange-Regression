import java.io._
import collection.mutable.ArrayBuffer
import org.supercsv.io.{CsvListReader, CsvListWriter}
import org.supercsv.prefs.CsvPreference

/* Feature Extraction Function
 * I assume here that a instance is represented by a List of Strings*/
class FEFun(val name: String, val fun: Array[String] => Array[String]);

/* Feature Extraction Pipeline */
class FEPipeline(funs: List[FEFun]) {

  def addFEFun(f: FEFun): Unit = {
    funs :+ f
  }

  /* print the names of the feature functions in the pipeline */
  def showFuns():Unit = {
    funs.zipWithIndex.map(t => println(t._2 + ". " +  t._1.name))
  }

  /* Take a row and transform it according to all of the feature functions */
  private def applyAll(row: Array[String]): Array[String] = {
    funs.flatMap(f => f.fun(row)).toArray
  }

  /* Implicit Sig: (CSVFile -> CSVFile)
   * apply all of the feature functions in the pipeline to the csvfile
   * and write the new csvfile back out
   */
  def processFile(filename: String): List[Array[String]] = {
    val rows = getRowsFromFile(filename).toList
    rows.map(row => applyAll(row))
  }

  /* Write out a CSV File
   * Implicit Sig: List[Array[String]] -> Unit where you pass a design matrix
   */
  def writeCSV(filename: String, rows: List[Array[String]]): Unit = {
    val listWriter = new CsvListWriter(new FileWriter(filename),
      CsvPreference.STANDARD_PREFERENCE)
    for (row <- rows) {
      listWriter.write(row.toList)
    }
  }

  /*  Comment Required */
  private def getRowsFromFile(fileName: String): Seq[Array[String]] = {
    val csvFile = new File(fileName)
    val br      = new BufferedReader(new InputStreamReader(new
					    FileInputStream(csvFile)))
    val reader  = new CsvListReader(br, CsvPreference.STANDARD_PREFERENCE)
    val rows    = collectWhileValue(null !=)(reader.read()).map(
					_.toArray.map(_.asInstanceOf[String]))
    rows.drop(1)
  }

  /* Comment Required */
  private def collectWhileValue[A: Manifest, B >: A]
  (cond: B => Boolean)(value: => A): Seq[A] = {
    val arr    = new ArrayBuffer[A]()
    var curVal = value
    while (cond(curVal)) { arr += curVal; curVal = value }
    arr.toSeq
  }
}
