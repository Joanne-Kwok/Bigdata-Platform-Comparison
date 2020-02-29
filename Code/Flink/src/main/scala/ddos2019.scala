import java.text.SimpleDateFormat

import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.classification.SVM
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.slf4j.LoggerFactory

object ddos2019 {

  case class DataFormat(f1: String, f2: String, f3: String, f4: String, f5: String, f6: String, f7: String, f8: String, f9: String, f10: String, f11: String, f12: String, f13: String, f14: String, f15: String, f16: String, f17: String, f18: String, f19: String, f20: String, f21: String, f22: String, f23: String, f24: String, f25: String, f26: String, f27: String, f28: String, f29: String, f30: String, f31: String, f32: String, f33: String, f34: String, f35: String, f36: String, f37: String, f38: String, f39: String, f40: String, f41: String, f42: String, f43: String, f44: String, f45: String, f46: String, f47: String, f48: String, f49: String, f50: String, f51: String, f52: String, f53: String, f54: String, f55: String, f56: String, f57: String, f58: String, f59: String, f60: String, f61: String, f62: String, f63: String, f64: String, f65: String, f66: String, f67: String, f68: String, f69: String, f70: String, f71: String, f72: String, f73: String, f74: String, f75: String, f76: String, f77: String, f78: String, f79: String, f80: String, f81: String, f82: String, f83: String, f84: String, l1: String)
  val df = new SimpleDateFormat("yyyy-dd-MM HH:mm:ss:SSSS")

  def main(args: Array[String]): Unit = {
    val env = ExecutionEnvironment.getExecutionEnvironment
    val trainPath = "hdfs:///user/ubuntu/Hadoop/nb/ddos2019/train/DDOS2019_train3.csv"
    val testPath = "hdfs:///user/ubuntu/Hadoop/nb/ddos2019/test/DDOS2019_test3.csv"

    println("Loading data! Current time is: " + df.format(System.currentTimeMillis()))
    val trainInput = env.readCsvFile[DataFormat](trainPath) // DataSet
    val testInput = env.readCsvFile[DataFormat](testPath) // DataSet

    //    val totalTrain = trainInput.count()
    //    val totalTest = testInput.count()

    val totalTrain = 1479255
    val totalTest = 633966

    // Format Dataset
    println("Reformat data! Current time is: " + df.format(System.currentTimeMillis()))
    val trainData = trainInput.rebalance()
      .map { tuple =>
        val F = tuple.productIterator.toList.dropRight(1) //dropRight(10) -p16-s2 Accuracy high,
        val feature = F.map(_.asInstanceOf[String].toDouble)
        val L = tuple.productIterator.toList.last
        val label = if (L == "BENIGN") -1.0 else 1.0
        LabeledVector(label, DenseVector(feature.toArray))
      }

    val testData = testInput
      .map { tuple =>
        val F = tuple.productIterator.toList.dropRight(1) // dropRight(3) original
        val feature = F.map(_.asInstanceOf[String].toDouble)
        val L = tuple.productIterator.toList.last
        val label = if (L == "BENIGN") -1.0 else 1.0
        LabeledVector(label, DenseVector(feature.toArray))
      }

    // Start Fitting
    val svm = SVM()
      //     .setBlocks(1)
      .setBlocks(env.getParallelism)
      .setIterations(100)
      .setRegularization(0.001)
      .setStepsize(0.1)
    //     .setRegularization(1.11)
    //     .setStepsize(0.01)
    //     .setOutputDecisionFunction(true)
    // .setThreshold(98435.6)
    //      .setSeed(10)

    println("Start Training! Current time is: " + df.format(System.currentTimeMillis()))
    val trainstart = System.currentTimeMillis()
    svm.fit(trainData)

    val trainend = System.currentTimeMillis()
    println("End Training! Current time is: " + df.format(System.currentTimeMillis()))
    // Start Predict
    val testD = testData.map(t => (t.vector, t.label))


    println("Start Prediction! Current time is: " + df.format(System.currentTimeMillis()))
    val predstart = System.currentTimeMillis()
    val predict: DataSet[(Double, Double)] = svm.evaluate(testD)
    val predend = System.currentTimeMillis()
    println("End Prediction! Current time is: " + df.format(System.currentTimeMillis()))

    //    predict.print()

    // Calculate Accuracy
    //    println("Calculating Accuracy! Current time is: " + df.format(System.currentTimeMillis()))
    println("Whole Job Execute! Current time is: " + df.format(System.currentTimeMillis()))
    val jobstart = System.currentTimeMillis()
    predict.map(t => (t._1, t._2, 1)).groupBy(0,1).reduce((x1,x2) => (x1._1, x1._2, x1._3 + x2._3)).print()
    val jobend = System.currentTimeMillis()
    println("Whole Job Finish! Current time is: " + df.format(System.currentTimeMillis()))
    /*
        val tp = predict.filter { e => e._1.toDouble == 1.0 && e._2.toDouble == 1.0}.count
        val tn = predict.filter { e => e._1.toDouble == -1.0 && e._2.toDouble == -1.0}.count
        val fp = predict.filter { e => e._1.toDouble == -1.0 && e._2.toDouble == 1.0}.count
        val fn = predict.filter { e => e._1.toDouble == 1.0 && e._2.toDouble == -1.0}.count

        val precision  = (tp.toDouble/(tp.toDouble + fp.toDouble)).formatted("%.6f").toDouble
        val recall = (tp.toDouble/(tp.toDouble + fn.toDouble)).formatted("%.6f").toDouble
    */
    // Print Output
    println("Job Finished! Current time is: " + df.format(System.currentTimeMillis()))
    println("************** Output ************************")
    println("Total Training cases = " + totalTrain.toInt )
    println("Total Test cases = " + totalTest.toInt )
    println(s"Training Time = ${trainend-trainstart} ms")
    println(s"Prediction Time = ${predend-predstart} ms" )
    println(s"Total Time = ${jobend-jobstart} ms" )
    /*    println("Accuracy = " + (tp.toDouble + tn.toDouble)/totalTest.toDouble)
        println("tp = " + tp.toInt )
        println("tn = " + tn.toInt )
        println("fp = " + fp.toInt )
        println("fn = " + fn.toInt )
        println("Detect Rate = " + precision)
        println("Recall = " + recall)
        println("F1 = " + 2.0/ ((1.0/precision)+(1.0/recall)))*/
    println("***********************************************")

  }

}
