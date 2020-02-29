import java.text.SimpleDateFormat

import com.sun.org.slf4j.internal.LoggerFactory
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.classification.SVM
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.metrics.Counter
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.slf4j.LoggerFactory

object kdd {

  class MyMapper extends RichMapFunction[String,String] {
    @transient private var counter: Counter = _

    override def open(parameters: Configuration): Unit = {
      counter = getRuntimeContext()
        .getMetricGroup()
        .counter("myCounter")
    }

    override def map(value: String): String = {
      counter.inc()
      value
    }
  }

  case class DataFormat1(f1: String, f2: String, f3: String, f4: String, f5: String, f6: String, f7: String, f8: String, f9: String, f10: String, f11: String, f12: String, f13: String, f14: String, f15: String, f16: String, f17: String, f18: String, f19: String, f20: String, f21: String, f22: String, f23: String, f24: String, f25: String, f26: String, f27: String, f28: String, f29: String, f30: String, f31: String, f32: String, f33: String, f34: String, f35: String, f36: String, f37: String, f38: String, f39: String, f40: String, f41: String, l1: String)


//  val logger = LoggerFactory.getLogger(this.getClass)
  val df1 = new SimpleDateFormat("yyyy-dd-MM HH:mm:ss:SSSS")

  def main(args: Array[String]): Unit = {

    val env = ExecutionEnvironment.getExecutionEnvironment
    //val trainPath = "hdfs:///user/hadoop/nb/kdd/train/KDDTr1.txt"
    //val testPath = "hdfs:///user/hadoop/nb/kdd/train/KDDTr1.txt"
    val trainPath = "hdfs:///user/ubuntu/Hadoop/nb/kdd/train/KDDTrain.txt"
    val testPath = "hdfs:///user/ubuntu/Hadoop/nb/kdd/test/KDDTest.txt"

    println("Loading data! Current time is: " + df1.format(System.currentTimeMillis()))
    val trainInput = env.readCsvFile[DataFormat1](trainPath) // DataSet
    val testInput = env.readCsvFile[DataFormat1](testPath) // DataSet


    //    val totalTrain = trainInput.count()
    //    val totalTest = testInput.count()
    val totalTrain = 494021
    val totalTest = 311029

    // Format Dataset
    println("Reformat data! Current time is: " + df1.format(System.currentTimeMillis()))
    val trainData = trainInput
      .map{tuple =>
        val F = tuple.productIterator.toList.dropRight(1)
        val feature = F.map(_.asInstanceOf[String].toDouble)
        val L = tuple.productIterator.toList.last
        val label = if (L=="normal.") -1 else 1
        LabeledVector(label, DenseVector(feature.toArray))
      }

    val testData = testInput
      .map{tuple =>
        val F = tuple.productIterator.toList.dropRight(1)
        val feature = F.map(_.asInstanceOf[String].toDouble)
        val L = tuple.productIterator.toList.last
        val label = if (L=="normal.") -1 else 1
        LabeledVector(label, DenseVector(feature.toArray))
      }

    // Start Fitting
    val svm = SVM()
      .setBlocks(env.getParallelism)
      .setIterations(100)
      .setRegularization(0.001)
      .setStepsize(0.1)

    println("Start Training! Current time is: " + df1.format(System.currentTimeMillis()))
    val trainstart = System.currentTimeMillis()
    svm.fit(trainData)
    val trainend = System.currentTimeMillis()
    println("End Training! Current time is: " + df1.format(System.currentTimeMillis()))
    // Start Predict
    val testD = testData.map(t => (t.vector, t.label))


    println("Start Prediction! Current time is: " + df1.format(System.currentTimeMillis()))
    val predstart = System.currentTimeMillis()
    val predict: DataSet[(Double, Double)]  =  svm.evaluate(testD)
    val predend = System.currentTimeMillis()
    println("End Prediction! Current time is: " + df1.format(System.currentTimeMillis()))

    //       predict.print()
    println("Whole Job Execute! Current time is: " + df1.format(System.currentTimeMillis()))
    val jobstart = System.currentTimeMillis()
    predict.map(t => (t._1, t._2, 1)).groupBy(0,1).reduce((x1,x2) => (x1._1, x1._2, x1._3 + x2._3)).print()
    val jobend = System.currentTimeMillis()
    println("Whole Job Finish! Current time is: " + df1.format(System.currentTimeMillis()))

    // Calculate Accuracy
    //    println("Calculating Accuracy! Current time is: " + df1.format(System.currentTimeMillis()))
    /*
        val tpI = predict.map(a => ( "fp",if(a._1.toDouble == 1.0 && a._2.toDouble == 1.0 ) 1.toInt else 0.toInt ) ).sum(1).setParallelism(30).collect()
        val tnI = predict.map(a => ( "tn",if(a._1.toDouble == -1.0 && a._2.toDouble == -1.0) 1.toInt else 0.toInt ) ).sum(1).setParallelism(30).collect()
        val fpI = predict.map(a => ( "fp",if(a._1.toDouble == -1.0 && a._2.toDouble == 1.0) 1.toInt else 0.toInt) ).sum(1).setParallelism(30).collect()
        val fnI = predict.map(a => ( "fn",if(a._1.toDouble == 1.0 && a._2.toDouble == -1.0) 1.toInt else 0.toInt) ).sum(1).setParallelism(30).collect()

        val tp = tpI.head._2
        val tn = tnI.head._2
        val fp = fpI.head._2
        val fn = fnI.head._2
    */
    //      println("new")

    /*
        val tp = predict.filter { e => e._1.toDouble == 1.0 && e._2.toDouble == 1.0}.setParallelism(30).count
        val tn = predict.filter { e => e._1.toDouble == -1.0 && e._2.toDouble == -1.0}.setParallelism(30).count
        val fp = predict.filter { e => e._1.toDouble == -1.0 && e._2.toDouble == 1.0}.setParallelism(30).count
        val fn = predict.filter { e => e._1.toDouble == 1.0 && e._2.toDouble == -1.0}.setParallelism(30).count
    */
    //    val precision  = (tp.toDouble/(tp.toDouble + fp.toDouble)).formatted("%.6f").toDouble
    //    val recall = (tp.toDouble/(tp.toDouble + fn.toDouble)).formatted("%.6f").toDouble

    // Print Output
    //   println("Job Finished! Current time is: " + df1.format(System.currentTimeMillis()))
    println("************** Output ************************")
    println("Total Training cases = " + totalTrain.toInt )
    println("Total Test cases = " + totalTest.toInt )
    println(s"Training Time = ${trainend-trainstart} ms")
    println(s"Prediction Time = ${predend-predstart} ms" )
    println(s"Total Time = ${jobend-jobstart} ms" )
    /*    println("Accuracy = " + (tp.toDouble + tn.toDouble)/totalTest.toDouble)
        println("tp = " + tp.toInt )
        println("tn  = " + tn.toInt )
        println("fp = " + fp.toInt )
        println("fn = " + fn.toInt )
        println("Detect Rate = " + precision)
        println("Recall = " + recall)
        println("F1 = " + 2.0/ ((1.0/precision)+(1.0/recall)))*/
    println("***********************************************")

  }
}

/*
Results (setParallelism(30)):
Starting execution of program
Loading data! Current time is: 2020-01-27 23:40:17:0810
Reformat data! Current time is: 2020-01-27 23:40:18:0858
Start Training! Current time is: 2020-01-27 23:40:18:0903
End Training! Current time is: 2020-01-27 23:40:19:0804
Start Prediction! Current time is: 2020-01-27 23:40:19:0807
End Prediction! Current time is: 2020-01-27 23:40:19:0834
Calculating Accuracy! Current time is: 2020-01-27 23:40:19:0834
new
Job Finished! Current time is: 2020-01-27 23:46:17:0450
************** Output ************************
Total Training cases = 494021
Total Test cases = 311029
Training Time = 901 ms
Prediction Time = 27 ms
Accuracy = 0.9015750942838128
tp = 237456
tn  = 42960
fp = 16340
fn = 13891
Detect Rate = 0.935618
Recall = 0.944734
F1 = 0.9401539026863054
***********************************************
Program execution finished
Job with JobID ecd7a774c8123c7a125cec565360061f has finished.
Job Runtime: 86143 ms
Accumulator Results:
- 6f2a37131f8fbcefa4273aa788697c0b (java.util.ArrayList) [1 elements]

 */