package seregression

object FeatEx {

  def main(args: Array[String]): Unit = {
    val ident = new FEFun("identity", (x:Array[String]) => x)
    val silly = new FEFun("Silly",    (x:Array[String]) => Array("1","2"))
    val pipe  = new FEPipeline(List(ident, silly))

    pipe.writeCSV("./test.csv", pipe.processFile(args(0)))
    pipe.showFuns()
  }
}
