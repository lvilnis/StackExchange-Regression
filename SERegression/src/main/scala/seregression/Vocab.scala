package seregression

import collection.mutable

class Vocab {

  private var vocab  = new mutable.HashMap[String, Int]
  private var counts = new mutable.HashMap[String, Int]
  private var unused = List[Int]()
  private var nextIndex = 0

  def add(key: String): Unit = {
    if (!vocab.contains(key)) vocab  += ((key, get_next_index()))
    counts += ((key, counts.getOrElse(key, 0) + 1))
  }

  private def get_next_index(): Int = {
    var index = -1             //place holder
    if (unused.length > 0) {
      index  = unused.head
      unused = unused.tail
    } else {
      index = nextIndex
      nextIndex += 1
    }
    index
  }

  def remove(key: String): Unit = {
    unused = unused :+ vocab.remove(key).get
    counts.remove(key)
  }

  def show():Unit = {
    vocab.foreach(e => println(e))
  }

  def size(): Int = {
    vocab.size
  }

  def index(token: String): Int = {
    vocab.getOrElse(token, -1)
  }

  def cull(thresh: Int = 1): Unit = {
    //maybe return list of indicies of culled tokens
    counts.foreach(e => if (e._2 <= thresh) remove(e._1))
  }
}
