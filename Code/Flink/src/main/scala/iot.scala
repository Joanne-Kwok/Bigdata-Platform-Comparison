import java.text.SimpleDateFormat

import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.classification.SVM
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.slf4j.LoggerFactory


object iot {

  case class DataFormat(f1: String, f2: String, f3: String, f4: String, f5: String, f6: String, f7: String, f8: String, f9: String, f10: String, f11: String, f12: String, f13: String, f14: String, f15: String, f16: String, f17: String, f18: String, f19: String, f20: String, f21: String, f22: String, f23: String, f24: String, f25: String, l1: String)


  val df = new SimpleDateFormat("yyyy-dd-MM HH:mm:ss:SSSS")

  def main(args: Array[String]): Unit = {
    val env = ExecutionEnvironment.getExecutionEnvironment
    val trainPath = "hdfs:///user/ubuntu/Hadoop/nb/Iot/train/IOT_train.csv"
    val testPath = "hdfs:///user/ubuntu/Hadoop/nb/Iot/test/IOT_test.csv"

    println("Loading data! Current time is: " + df.format(System.currentTimeMillis()))
    val trainInput = env.readCsvFile[DataFormat](trainPath) // DataSet
    val testInput = env.readCsvFile[DataFormat](testPath) // DataSet
    val totalTrain = trainInput.count()
    val totalTest = testInput.count()

    // Format Dataset
    println("Reformat data! Current time is: " + df.format(System.currentTimeMillis()))
    val trainData = trainInput
      .map { tuple =>
        val F = tuple.productIterator.toList.dropRight(1)
        val feature = F.map(_.asInstanceOf[String].toDouble)
        val L = tuple.productIterator.toList.last.asInstanceOf[String].toDouble
        val label = if (L == 0.0) -1.0 else 1.0
        LabeledVector(label, DenseVector(feature.toArray))
      }

    val testData = testInput
      .map { tuple =>
        val F = tuple.productIterator.toList.dropRight(1)
        val feature = F.map(_.asInstanceOf[String].toDouble)
        val L = tuple.productIterator.toList.last.asInstanceOf[String].toDouble
        val label = if (L == 0.0) -1.0 else 1.0
        LabeledVector(label, DenseVector(feature.toArray))
      }

    // Start Fitting
    val svm = SVM()
      .setBlocks(env.getParallelism)
      .setIterations(100)
      .setRegularization(0.001)
      .setStepsize(0.1)

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
    
    println("Whole Job Execute! Current time is: " + df.format(System.currentTimeMillis()))
    val jobstart = System.currentTimeMillis()
    predict.map(t => (t._1, t._2, 1)).groupBy(0,1).reduce((x1,x2) => (x1._1, x1._2, x1._3 + x2._3)).print()
    val jobend = System.currentTimeMillis()
    println("Whole Job Finish! Current time is: " + df.format(System.currentTimeMillis()))

    // Print Output
    println("Job Finished! Current time is: " + df.format(System.currentTimeMillis()))
    println("************** Output ************************")
    println("Total Training cases = " + totalTrain.toInt )
    println("Total Test cases = " + totalTest.toInt )
    println(s"Training Time = ${trainend-trainstart} ms")
    println(s"Prediction Time = ${predend-predstart} ms" )
    println(s"Total Time = ${jobend-jobstart} ms" )
    println("***********************************************")

  }

}
