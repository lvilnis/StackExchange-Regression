package seregression

import java.io._
import collection.mutable.ArrayBuffer
import cc.factorie._
import cc.factorie.app.classify._
import la.{Tensor, Tensor1}
import optimize._
import org.supercsv.io.CsvListReader
import org.supercsv.prefs.CsvPreference
import edu.stanford.nlp.process.{CoreLabelTokenFactory, PTBTokenizer}
import edu.stanford.nlp.ling.CoreLabel

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
//  val splitRegex = "\\w+".r
//  def tokenize(body: String): Seq[String] = splitRegex.findAllIn(body).toSeq

  // todo: write regexes to strip html and also add special features when html is removed like "#CodeTag#", "#PreTag"#, "#BlockquoteTag#", etc
  def tokenize(body: String): Seq[String] = {
    val tokenizer = new PTBTokenizer[CoreLabel](new StringReader(body), new CoreLabelTokenFactory, "")
    val output = collectWhile((_: String) => tokenizer.hasNext)(tokenizer.next().value)
//    output.foreach(println(_))
    output
  }

  def getRowsFromFile(fileName: String): Seq[Array[String]] = {
    val csvFile = new File(fileName)
    val br = new BufferedReader(new InputStreamReader(new FileInputStream(csvFile)))
    val reader = new CsvListReader(br, CsvPreference.STANDARD_PREFERENCE)
    val rows = collectWhile(null !=)(reader.read()).map(_.toArray.map(_.asInstanceOf[String]))
    rows.drop(1)
  }

  // todo: use factories TUI args classes for this to add nice options like for stoplists?
  def main(rawArgs: Array[String]): Unit = {

    val args = if (rawArgs.isEmpty) Array(lukesPath, lukesPath2) else rawArgs

    // shuffle instances and take 10K for now
    val rows = args.toSeq.flatMap(getRowsFromFile(_)).shuffle.take(30000)

    def cell(row: Array[String], c: String): String = row(col(c) - 1)

    val instances = new ArrayBuffer[(Boolean, Int, Seq[String])]
    for (r <- rows) {
      val bodyStr = cell(r, "Body")
      val closedDateStr = cell(r, "ClosedDate")
      val idStr = cell(r, "Id")
      val tokens = tokenize(bodyStr)
      val id = idStr.toInt
      val isClosed = closedDateStr != null
      instances += ((isClosed, id, tokens))
    }

    object FeaturesDomain extends CategoricalTensorDomain[String]
    object LabelDomain extends CategoricalDomain[String]

    val labels = new LabelList[Label, Features](_.features)

    for ((isClosed, id, tokens) <- instances) {
      val f = new BinaryFeatures(if (isClosed) "Closed" else "Open", id.toString, FeaturesDomain, LabelDomain)
      // gotta remove the "duplicate" feature or else we're gonna think we're smart when we're not!!
      // interesting that things like "<code>" tags have high info-gain - I guess people who dont bother to even include
      // code samples get closed more often? this means we shouldn't just strip out all the tags, we should at least add
      // features like "#HasCodeTags#" and whatnot.
      tokens.map(_.toLowerCase).filter("duplicate" !=).foreach(f +=)
      labels += f.label
      labels.instanceWeight(f.label) = if (isClosed) 1.0 else 1.0
    }

    val (trlabels, tslabels) = labels.shuffle.split(0.7)
    val trainLabels = new LabelList[Label, Features](trlabels, _.features)
    val testLabels = new LabelList[Label, Features](tslabels, _.features)

    println("Read " + labels.length + " instances with " + instances.filterNot(_._1).length + " closed questions.")

    println("Vocabulary size: " + FeaturesDomain.dimensionDomain.size)

    println("Top 40 features with highest information gain: ")
    new InfoGain(labels).top(40).foreach(println(_))

    val model = new LogLinearModel[Label, Features](_.features, LabelDomain, FeaturesDomain)
    val classifier = new ModelBasedClassifier[Label](model, LabelDomain)

//    trainModelSVMSGD(trainLabels, model)
//    trainModelLogisticRegression(trainLabels, model)
    // this one's pretty much the best right now and fast
    trainModelLogisticRegressionSGD(trainLabels, model)
//    trainModelLibLinearSVM(trainLabels, model)
    // ARG stupid naive bayes being almost as good as logistic reg and SVM
//    trainModelNaiveBayes(trainLabels, model)

    printTrial("== Training Evaluation ==", trainLabels, classifier)
    printTrial("== Testing Evaluation ==", testLabels, classifier)
  }

  def trainModelLogisticRegression(labels: LabelList[Label, Features], model: LogLinearModel[Label, Features]) = {
    val lbfgs = new LBFGS with L2Regularization { variance = 10.0 }
    val strategy = new BatchTrainer(model, lbfgs)
    trainModel(labels, _ => strategy.isConverged, strategy, ObjectiveFunctions.logMultiClassObjective)
  }

  def trainModelLogisticRegressionSGD(labels: LabelList[Label, Features], model: LogLinearModel[Label, Features]) = {
    val strategy = new InlineSGDTrainer(model, optimizer = new AdagradAccumulatorMaximizer(model))
    trainModel(labels, 5 <, strategy, ObjectiveFunctions.logMultiClassObjective)
  }

  def trainModelLibLinearSVM(ll: LabelList[Label, Features], model: LogLinearModel[Label, Features]) = {
    val xs: Seq[Tensor1] = ll.map(ll.labelToFeatures(_).tensor.asInstanceOf[Tensor1])
    val ys: Array[Int]   = ll.map(_.intValue).toArray
    val weightTensor = for (label <- 0 until 2) yield new LinearL2SVM().train(xs, ys, label)
    for (f <- 0 until ll.featureDomain.size) {
      model.evidenceTemplate.weights(0, f) = weightTensor(0)(f)
      model.evidenceTemplate.weights(1, f) = weightTensor(1)(f)
    }
  }

  def trainModelSVMSGD(labels: LabelList[Label, Features], model: LogLinearModel[Label, Features]) = {
    val strategy = new InlineSGDTrainer(model, optimizer = new AdagradAccumulatorMaximizer(model))
    trainModel(labels, 5 <, strategy, ObjectiveFunctions.hingeMultiClassObjective)
  }

  def trainModelNaiveBayes(labels: LabelList[Label, Features], model: LogLinearModel[Label, Features]) = {
    val newModel = new NaiveBayesTrainer().train(labels).asInstanceOf[ModelBasedClassifier[Label]].model.asInstanceOf[LogLinearModel[Label, Features]]
    def copyWeights(to: Tensor, from: Tensor): Unit = from.foreachActiveElement((i, v) => to(i) = v)
    copyWeights(model.evidenceTemplate.weights, newModel.evidenceTemplate.weights)
    copyWeights(model.biasTemplate.weights, newModel.biasTemplate.weights)
  }

  def trainModel(labels: LabelList[Label, Features], isConverged: Int => Boolean,
    strategy: Trainer[LogLinearModel[Label, Features]], obj: ObjectiveFunctions.MultiClassObjectiveFunction) = {
    val pieces = labels.toIterable.map(l => new GLMExample(
      labels.labelToFeatures(l).tensor.asInstanceOf[Tensor1],
      l.intValue, obj, weight = labels.instanceWeight(l)))
    val start = System.currentTimeMillis
    var i = 0
    while (!isConverged(i)) {
      strategy.processExamples(pieces)
      i += 1
    }
    println("Classifier trained in " + ((System.currentTimeMillis - start) / 1000.0) + " seconds.")
  }

  def printTrial(label: String, labels: Iterable[Label], classifier: Classifier[Label]): Unit = {
    val testTrial = new Trial[Label](classifier)
    testTrial ++= labels
    println(label)
    println(testTrial.toString)
  }

  def collectWhile[A: Manifest, B >: A](cond: B => Boolean)(value: => A): Seq[A] = {
    val arr = new ArrayBuffer[A]()
    var curVal = value
    while (cond(curVal)) { arr += curVal; curVal = value }
    arr.toSeq
  }

}