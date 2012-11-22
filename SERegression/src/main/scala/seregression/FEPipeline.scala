/* Feature Extraction Function
 * I assume here that a instance is represented by a List of Strings*/
class FEFun(val name: String, val fun: List[String] => List[String]);

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
  private def applyAll(row: List[String]): List[String] = {
    funs.flatMap(f => f.fun(row))
  }

  /* Implicit Sig: (CSVFile -> CSVFile)
   * apply all of the feature functions in the pipeline to the csvfile
   * and write the new csvfile back out
   */
  def processFile(filename: String): Unit = {
    // val csvFile = readCSV(filename)
    val rows = List[List[String]]()
    rows.map(row => applyAll(row))
  }
}
