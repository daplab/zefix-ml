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

import org.apache.spark.mllib.linalg.distributed.{RowMatrix, IndexedRowMatrix, IndexedRow, BlockMatrix}

import org.apache.log4j.Logger
import org.apache.log4j.Level

Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

val sameModel = DistributedLDAModel.load(sc, "hdfs://daplab2/user/acholleton/zefix-ml/ldaModel")
val topicDistributions: RDD[(Long, Vector)] = sameModel.topicDistributions

val corpus = sc.textFile("hdfs://daplab2/user/acholleton/zefix-ml/corpus")

// Close your eyes =>
val mat = new IndexedRowMatrix(topicDistributions.map(l => new IndexedRow(l._1, l._2))).toBlockMatrix.transpose.toIndexedRowMatrix.toRowMatrix


val sim = new RowMatrix(topicDistributions.map(_._2))
val sim = mat.columnSimilarities
