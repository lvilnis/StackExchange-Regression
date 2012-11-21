package seregression

import java.io.{InputStreamReader, BufferedReader, FileInputStream, File}
import collection.mutable.ArrayBuffer
import cc.factorie._
import cc.factorie.app.classify._
import la.Tensor1
import optimize._
import org.supercsv.io.CsvListReader
import org.supercsv.prefs.CsvPreference

object Test {

  val col = Map(
    "Id" -> 1,
    "PostTypeId" -> 2,
    "AcceptedAnswerId" -> 3,
    "ParentId" -> 4,
    "CreationDate" -> 5,
    "Score" -> 6,
    "ViewCount" -> 7,
    "Body" -> 8,
    "OwnerUserId" -> 9,
    "OwnerDisplayName" -> 10,
    "LastEditorUserId" -> 11,
    "LastEditorDisplayName" -> 12,
    "LastEditDate" -> 13,
    "LastActivityDate" -> 14,
    "Title" -> 15,
    "Tags" -> 16,
    "AnswerCount" -> 17,
    "CommentCount" -> 18,
    "FavoriteCount" -> 19,
    "ClosedDate" -> 20,
    "CommunityOwnedDate" -> 21)

  val lukesPath = """C:\Users\Luke\Dropbox\MLFinalProj (1)\data\postTypeId=1_closed_is_null_creation_gt_2011-08-13.csv"""
  val lukesPath2 = """C:\Users\Luke\Dropbox\MLFinalProj (1)\data\postTypeId=1_closed_gt_2011-08-13_creation_gt_2011-08-13.csv"""

  // todo: remove html
  val splitRegex = "\\w+".r
  def tokenize(body: String): Seq[String] = splitRegex.findAllIn(body).toSeq

  def getRowsFromFile(fileName: String): Seq[Array[String]] = {
    val csvFile = new File(fileName)
    val br = new BufferedReader(new InputStreamReader(new FileInputStream(csvFile)))
    val reader = new CsvListReader(br, CsvPreference.STANDARD_PREFERENCE)
    val rows = collectWhile(null !=)(reader.read()).map(a => { a.toArray.map(_.asInstanceOf[String]) })
    rows.drop(1)
  }

  // todo: use factories TUI args classes for this to add nice options like for stoplists?
  def main(rawArgs: Array[String]): Unit = {

    val args = if (rawArgs.isEmpty) Array(lukesPath, lukesPath2) else rawArgs

    // shuffle instances and take 10K for now
    val rows = args.toSeq.flatMap(getRowsFromFile(_)).shuffle.take(10000)

    def cell(row: Array[String], c: String): String = row(col(c) - 1)

    val instances = new ArrayBuffer[(Boolean, Int, Seq[String])]
    rows.foreach(r => {
      val bodyStr = cell(r, "Body")
      val closedDateStr = cell(r, "ClosedDate")
      val idStr = cell(r, "Id")
      val tokens = tokenize(bodyStr)
      val id = idStr.toInt
      val isClosed = closedDateStr != null
      instances += ((isClosed, id, tokens))
    })

    object FeaturesDomain extends CategoricalTensorDomain[String]
    object LabelDomain extends CategoricalDomain[String]

    val labels = new LabelList[Label, Features](_.features)

    for ((isClosed, id, tokens) <- instances) {
      val f = new BinaryFeatures(if (isClosed) "Closed" else "Open", id.toString, FeaturesDomain, LabelDomain)
      f ++= tokens
      labels += f.label
      labels.instanceWeight(f.label) = if (isClosed) 1.0 else 1.0
    }

    println("Read " + labels.length + " instances with " + instances.filterNot(_._1).length + " closed questions.")

    val model = new LogLinearModel[Label, Features](_.features, LabelDomain, FeaturesDomain)
    val classifier = new ModelBasedClassifier[Label](model, LabelDomain)

    val pieces = labels.map(l => new GLMExample(
      labels.labelToFeatures(l).tensor.asInstanceOf[Tensor1],
      l.intValue,
      ObjectiveFunctions.logMultiClassObjective,
      weight = labels.instanceWeight(l)))

    val lbfgs = new optimize.LBFGS with L2Regularization { variance = 10.0 }
    val strategy = new BatchTrainer(model, lbfgs)

    while (!strategy.isConverged)
      strategy.processExamples(pieces)

    val trainTrial = new Trial[Label](classifier)
    trainTrial ++= labels
    println("== Training Evaluation ==")
    println(trainTrial.toString)

  }

  def collectWhile[A: Manifest, B >: A](cond: B => Boolean)(value: => A): Seq[A] = {
    val arr = new ArrayBuffer[A]()
    var curVal = value
    while (cond(curVal)) { arr += curVal; curVal = value }
    arr.toSeq
  }

}
