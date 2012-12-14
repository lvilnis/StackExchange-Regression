package seregression

import java.io._
import collection.mutable.ArrayBuffer
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.process.{CoreLabelTokenFactory, PTBTokenizer}

object FeatEx {

  val col = Map(
    "PostId" -> 1,
    "PostCreationDate" -> 2,
    "OwnerUserId" -> 3,
    "OwnerCreationDate" -> 4,
    "ReputationAtPostCreation" -> 5,
    "OwnerUndeletedAnswerCountAtPostTime" -> 6,
    "Title" -> 7,
    "BodyMarkdown" -> 8,
    "Tag1" -> 9,
    "Tag2" -> 10,
    "Tag3" -> 11,
    "Tag4" -> 12,
    "Tag5" -> 13,
    "PostClosedDate" -> 14,
    "OpenStatus" -> 15)

  def main(args: Array[String]): Unit = {
    val ident = new FEFun("identity", (x:Array[String]) => x)
    val pipe  = new FEPipeline(List(ident))

    //pipe.writeCSV("./test.csv", pipe.processFile(args(0)))
    val rows = pipe.getRowsFromFile(args(0))
    val text = rows(0)(7)
    val tokenizer = new PTBTokenizer[CoreLabel](new StringReader(text), new CoreLabelTokenFactory, "")

    val vocab = new Vocab
    var quest = List[Int]()
    while(tokenizer.hasNext) {
      val token = tokenizer.next().value
      vocab.add(token)
      quest = quest :+ vocab.index(token)
    }
    println(quest)

  }
}
