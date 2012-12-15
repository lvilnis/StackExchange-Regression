package seregression

import java.io._
import collection.mutable.ArrayBuffer
import cc.factorie._
import cc.factorie.app.classify._
import la.{Tensor, Tensor1}
import optimize._
import org.supercsv.io.CsvListReader
import org.supercsv.io.CsvListWriter
import org.supercsv.prefs.CsvPreference
import edu.stanford.nlp.process.{CoreLabelTokenFactory, PTBTokenizer}
import edu.stanford.nlp.ling.CoreLabel
import scala.util.Random
import scala.util.matching.Regex
import collection.mutable

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
    "CommunityOwnedDate" -> 21,
    "Reputation" -> 22)

//  val col = Map(
//    "PostId" -> 1,
//    "PostCreationDate" -> 2,
//    "OwnerUserId" -> 3,
//    "OwnerCreationDate" -> 4,
//    "ReputationAtPostCreation" -> 5,
//    "OwnerUndeletedAnswerCountAtPostTime" -> 6,
//    "Title" -> 7,
//    "BodyMarkdown" -> 8,
//    "Tag1" -> 9,
//    "Tag2" -> 10,
//    "Tag3" -> 11,
//    "Tag4" -> 12,
//    "Tag5" -> 13,
//    "PostClosedDate" -> 14,
//    "OpenStatus" -> 15)

  val lukesPath = """C:\Users\Luke\Dropbox\MLFinalProj (1)\data\postTypeId=1_closed_is_null_creation_gt_2011-08-13.csv"""
  val lukesPath2 = """C:\Users\Luke\Dropbox\MLFinalProj (1)\data\postTypeId=1_closed_gt_2011-08-13_creation_gt_2011-08-13.csv"""

  // "strong" feature might be cheating too since it indicates moderator edits. need to write a regex to get rid of
  // moderator edits
  // ugh and having urls in your query will be highly correlated to duplicate too since possible duplicate flags include links
  // need to strip off the whole duplication warning
  val possibleDuplicateRegex = "(<strong>\\w*(P|p)ossible.*?</strong>)|((P|p)ossible (d|D)uplicate:)|((C|c)losed)".r
  val urlRegex = "<a.*?</a>".r
  val codeRegex = "(?s)(<code>.*?</code>)|(<pre>.*?</pre>)".r
  val quoteRegex = "(?s)<blockquote>.*?</blockquote>".r
  val imgRegex = "<img>".r
  val tagRegex = "(<[A-Za-z]+>)|(<[A-Za-z]+/>)|(</[A-Za-z]+>)".r
  val wsRegex = "\\s+".r

  case class Replacement(regex: Regex, replaceWith: String, feature: String)
  val replacements = List(
    Replacement(urlRegex, "", "#URL#"),
    Replacement(codeRegex, "", "#CODE#"),
    Replacement(possibleDuplicateRegex, "", ""),
    Replacement(quoteRegex, "", "#BLOCKQUOTE#"),
    Replacement(imgRegex, "", "#IMG#"),
    Replacement(tagRegex, "", ""),
    Replacement(wsRegex, " ", ""))

  def tagsToFeatures(tags: String): Array[String] = tags.drop(1).dropRight(1).split("><")

  def bucketizeBodyLength(len: Int): String =
    if (len < 100) "0-100"
    else if (len < 500) "100-500"
    else if (len < 1000) "500-1000"
    else if (len < 2000) "1000-2000"
    else if (len < 5000) "2000-5000"
    else "5000+"

  def bucketizeReputation(rep: Int): String = {
    if (rep < 10) "0-10"
    else if (rep < 50) "10-50"
    else if (rep < 200) "50-200"
    else if (rep < 1000) "200-1000"
    else "1000+"
  }

  // todo: write regexes to strip html and also add special features when html is removed like "#CodeTag#", "#PreTag"#, "#BlockquoteTag#", etc
  def tokenize(body: String): (Seq[String], Seq[String]) = {
    val feats = new ArrayBuffer[String]
    feats += "#BodyLength" + bucketizeBodyLength(body.length) + "#"
    val replaced = replacements.foldLeft(body)({ case (b, Replacement(regex, repl, feat)) =>
      if (feat != "") feats += feat
      regex.replaceAllIn(b, repl)
    })
    val tokenizer = new PTBTokenizer[CoreLabel](new StringReader(replaced), new CoreLabelTokenFactory, "")
    val output = collectWhile(tokenizer.hasNext)(tokenizer.next().value)
//    output.foreach(println(_))
    (output, feats.toSeq)
  }

  def getRowsFromFile(fileName: String): Seq[Array[String]] = {
    val csvFile = new File(fileName)
    val br = new BufferedReader(new InputStreamReader(new FileInputStream(csvFile)))
    val reader = new CsvListReader(br, CsvPreference.STANDARD_PREFERENCE)
    val rows = collectWhileValue(null !=)(reader.read()).map(_.toArray.map(_.asInstanceOf[String]))
    rows.drop(1)
  }

  case class Speriment(technique: String, numInstances: Int, f1Closed: Double, f1Open: Double)

  def writeResultsToFile(fileName: String, exps: Iterable[Speriment]): Unit = {
    val csvFile = new File(fileName)
    val bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(csvFile)))
    val writer = new CsvListWriter(bw, CsvPreference.STANDARD_PREFERENCE)
    writer.write("Classifier", "NumInstances", "F1 Closed", "F1 Open")
    for (Speriment(t, n, f1c, f1o) <- exps)
      writer.write(t.toString, n.toString, f1c.toString, f1o.toString)
    writer.flush()
    bw.close()
  }

  // todo: use factories TUI args classes for this to add nice options like for stoplists?
  def main(rawArgs: Array[String]): Unit = {

    val args = if (rawArgs.isEmpty) Array(lukesPath, lukesPath2) else rawArgs

    // shuffle instances and take 10K for now
    val rows = args.toSeq.flatMap(getRowsFromFile(_)).shuffle(new Random(42)).take(200000)

    def cell(row: Array[String], c: String): String = row(col(c) - 1)

    val instances = new ArrayBuffer[(Boolean, Boolean, Int, Seq[String], Seq[String])]
    for (r <- rows) {
//      val bodyStr = cell(r, "BodyMarkdown")
      val bodyStr = cell(r, "Body")
//      val closedDateStr = cell(r, "OpenStatus")
      val closedDateStr = cell(r, "ClosedDate")
//      val idStr = cell(r, "PostId")
      val idStr = cell(r, "Id")
      val tagsStr = cell(r, "Tags")
      val titleStr = cell(r, "Title")
      val repStr = cell(r, "Reputation")
      val extraFeatures = new mutable.ArrayBuffer[String]
      extraFeatures ++= titleStr.split("\\w+").map("#Title-" + _ + "#")
      extraFeatures ++= tagsToFeatures(tagsStr).map("#Tag-" + _ + "#")
      extraFeatures += "#Reputation" + bucketizeReputation(repStr.toInt) + "#"
      val possibleDuplicate = possibleDuplicateRegex.findFirstIn(bodyStr).isDefined
      val (tokens, fts) = tokenize(bodyStr)
      extraFeatures ++= fts
//      extraFeatures ++= List("Tag1","Tag2","Tag3","Tag4","Tag5").map(cell(r,_)).map("#Tag-" + _ + "#")
      val id = idStr.toInt
      val isClosed = closedDateStr != null
      instances += ((isClosed, possibleDuplicate, id, tokens, extraFeatures))
    }

    object FeaturesDomain extends CategoricalTensorDomain[String] { dimensionDomain.gatherCounts = true }
    object LabelDomain extends CategoricalDomain[String]

    val possibleDuplicates = new mutable.HashSet[Label]()
    val docCounts = new mutable.HashMap[String, Int]()
    val labels = new LabelList[Label, Features](_.features)
    for (i <- 1 to 2) {
      var curInstance = 0
      for ((isClosed, possibleDuplicate, id, tokens, extraFeatures) <- instances) {
        curInstance += 1
        val unigrams = tokens.map(_.toLowerCase)
        // nice, can run on my laptop with 1-2-3 grams giving 2.2 million features
        def grams(i: Int) = unigrams.sliding(i).map(_.mkString(":"))

        val f = new NonBinaryFeatures(if (isClosed) "Closed" else "Open", id.toString, FeaturesDomain, LabelDomain) {
          override val skipNonCategories = true
        }

        def appearsMoreThanOnce(feat: String): Boolean = FeaturesDomain.dimensionDomain.count(feat) > 1

        if (i == 2) {
          val unigramCounts = unigrams
            .filter(appearsMoreThanOnce)
            .groupBy(identity)
            .mapValues(_.length)
          if (unigramCounts.size > 0) {
            val len = unigrams.length: Double
            val numDocs = instances.length: Double
            val tf = unigramCounts.mapValues(_ / len)
            val idf = unigramCounts.map(t => {
              t._1 -> math.log(numDocs / docCounts(t._1))
            })
            val maxtf = tf.values.max
            val tfidf = unigramCounts.map({ case (k, _) => k -> tf(k) / maxtf * idf(k) })
//            unigramCounts.keys.foreach(k => f += (k + "#TFIDF#", tfidf(k)))
            unigramCounts.keys.foreach(k => f += (k, 1))
            f += ("#AVGTFIDF#", tfidf.values.sum / tfidf.size)
          }
          if (curInstance % 10000 == 0) println("Processed " + curInstance + " instances.")
        } else {
          unigrams.distinct.foreach(u => {
            docCounts(u) = docCounts.getOrElse(u, 0) + 1
            f += u
          })
        }
//        grams(2).foreach(f +=)
//        grams(3).foreach(f +=)
        extraFeatures.distinct.foreach(f +=)

        // try some cross products of tags and word features
        val tags = extraFeatures.distinct.filter(_.startsWith("#Tag"))
        unigrams.distinct
          .flatMap(u => tags.map(ef => "#Pair-" + u + ":" + ef + "#"))
          .filter(f => i == 1 || appearsMoreThanOnce(f))
          .foreach(f +=)

        // sanity check
//        if (isClosed) f += "#GROUNDTRUTH#"

        labels += f.label
        labels.instanceWeight(f.label) = if (isClosed) 1.0 else 1.0
        if (possibleDuplicate) possibleDuplicates += f.label
      }
      if (i == 1) {
        // TODO just add some extra features for like # of uncommon words, etc - don't trim
//        FeaturesDomain.dimensionDomain.trimBelowCount(2)
        labels.remove(0, labels.length)
      }
    }

    docCounts.clear()

    val results = new mutable.HashSet[Speriment]()
    for (numInstances <- 200000 to labels.length by 20000) {
      val usedLabels = labels.take(numInstances)
      val (trlabels, tslabels) = usedLabels.shuffle(new Random(42)).split(0.7)
      val trainLabels = new LabelList[Label, Features](trlabels.filterNot(possibleDuplicates), _.features)
      val testLabels = new LabelList[Label, Features](tslabels.filterNot(possibleDuplicates), _.features)

      println("Read " + usedLabels.length + " instances with " + usedLabels.filterNot(_.intValue == 0).length + " closed questions.")

      println("Discarded " + usedLabels.filter(possibleDuplicates).size + " possible duplicates from data set.")

      println("Vocabulary size: " + FeaturesDomain.dimensionDomain.size)

      // in addition to information gain I want to print out the way the feature changes the distribution
      // so like if the orig dist is [0.5, 0.5] and the new dists are [0.25, 0.75] and [0.75, 0.25], I want to see
      // [0.5, -0.5] or something
      println("Top 40 features with highest information gain: ")
      new InfoGain(labels).top(40).foreach(println(_))

      val trainers = Map[String, (LabelList[Label, Features], LogLinearModel[Label, Features]) => Unit](
        "Hinge Loss w/ AdaGrad" -> trainModelSVMSGD
      , "Log Loss w/ AdaGrad" -> trainModelLogisticRegressionSGD
      , "Naive Bayes" -> trainModelNaiveBayes
      , "Liblinear SVM" -> trainModelLibLinearSVM
  //    , "L2 Logistic Regression" -> trainModelLogisticRegression
      )

      for ((trainerName, trainer) <- trainers) yield {
        println("=== " + trainerName + " ===")
        val model = new LogLinearModel[Label, Features](_.features, LabelDomain, FeaturesDomain)
        val classifier = new ModelBasedClassifier[Label](model, LabelDomain)

        val start = System.currentTimeMillis
        trainer(trainLabels, model)

        println("Classifier trained in " + ((System.currentTimeMillis - start) / 1000.0) + " seconds.")

        printTrial("== Training Evaluation ==", trainLabels, classifier)
        val (f1Closed, f1Open) = printTrial("== Testing Evaluation ==", testLabels, classifier)
        results += Speriment(trainerName, trainLabels.length + testLabels.length, f1Closed, f1Open)
      }
    }

    writeResultsToFile("C:\\experiment-results.csv", results)
  }

  def trainModelLogisticRegression(labels: LabelList[Label, Features], model: LogLinearModel[Label, Features]) = {
    val lbfgs = new LBFGS with L2Regularization { variance = 1.0 }
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
    val weightTensor = new LinearL2SVM().train(xs, ys, 0)
    for (f <- 0 until ll.featureDomain.size) {
      model.evidenceTemplate.weights(0, f) = weightTensor(f)
      model.evidenceTemplate.weights(1, f) = -weightTensor(f)
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
    var i = 0
    while (!isConverged(i)) {
      strategy.processExamples(pieces)
      i += 1
    }
  }

  def printTrial(label: String, labels: Iterable[Label], classifier: Classifier[Label]): (Double, Double) = {
    val origSettings = labels.map(l => l -> l.intValue).toMap
    val testTrial = new Trial[Label](classifier)
    testTrial ++= labels
    println(label)
    println(testTrial.toString)
    val results = (testTrial.f1(0), testTrial.f1(1))
    labels.foreach(l => l.set(origSettings(l))(null))
    results
  }

  def collectWhile[A: Manifest](cond: => Boolean)(value: => A): Seq[A] = {
    val arr = new ArrayBuffer[A]()
    while (cond) { arr += value }
    arr.toSeq
  }

  def collectWhileValue[A: Manifest, B >: A](cond: B => Boolean)(value: => A): Seq[A] = {
    val arr = new ArrayBuffer[A]()
    var curVal = value
    while (cond(curVal)) { arr += curVal; curVal = value }
    arr.toSeq
  }

}
