package org.daplab.zefix.lda

import org.apache.avro.file.DataFileReader
import org.apache.avro.file.DataFileWriter
import org.apache.avro.io.DatumReader
import org.apache.avro.io.DatumWriter
import org.apache.avro.specific.SpecificDatumReader
import org.apache.avro.mapreduce.AvroKeyInputFormat
import org.apache.avro.mapred.AvroKey
import org.apache.hadoop.io.NullWritable
import org.apache.avro.mapred.AvroInputFormat
import org.apache.avro.mapred.AvroWrapper
import org.apache.avro.generic.GenericRecord
import org.apache.avro.mapred.{AvroInputFormat, AvroWrapper}
import org.apache.hadoop.io.NullWritable
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD;

import org.apache.log4j.Logger
import org.apache.log4j.Level

Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)


object topics {

  def main(args: Array[String]) {
    lazy val sc = {
      val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
      new SparkContext(conf)
    }

    val path = "hdfs://daplab2/shared/zefix/company/2015/**/**/*.avro"
    val avroRDD = sc.hadoopFile[AvroWrapper[GenericRecord], NullWritable, AvroInputFormat[GenericRecord]](path)

    val corpus: RDD[(String, String)] = avroRDD.map(l =>
      (
        new String(l._1.datum.get("name").toString()),
        new String(l._1.datum.get("purpose").toString())
      )
    ).filter(_._2 contains " de ")


    corpus.collect().foreach(println)

    val tokenized: RDD[Seq[String]] = corpus.map(_._2.toLowerCase.split("\\s"))
      .map(_.filter(_.length > 3).filter(_.forall(java.lang.Character.isLetter)))

    // Choose the vocabulary.
    //   termCounts: Sorted list of (term, termCount) pairs
    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    //   vocabArray: Chosen vocab (removing common terms)

    val stopWords = Array("de", "des", "d", "un", "une", "le", "la", "les", "leur", "leurs", "mon", "ton", "son", "pour", "dans", "toute", "tout", "tous", "toutes", "ainsi", "je", "elle", "il", "tu", "nous", "vous", "ils", "elles", "sous", "sont", "a", "ont", "on", "autres", "autre")
    val numStopwords = 20

    val vocabArray: Array[String] =
      termCounts.takeRight(termCounts.size - numStopwords).map(_._1).filter(!stopWords.contains(_))

    //   vocab: Map term -> term index
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    // Convert documents into term count vectors
    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        //      val _vocab = broadcastVocab.value
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }

    // Set LDA parameters
    val numTopics = 30
    val lda = new LDA().setK(numTopics).setMaxIterations(100)

    val ldaModel = lda.run(documents)

    // Print topics, showing top-weighted 10 terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)

    topicIndices.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabArray(term.toInt)}\t$weight")
      }
      println()
    }

    corpus.saveAsTextFile("hdfs://daplab2/user/acholleton/zefix-ml/corpus")
    ldaModel.save(sc, "hdfs://daplab2/user/acholleton/zefix-ml/ldaModel")

  }
}
