// Databricks notebook source
// MAGIC %md
// MAGIC Copyright 2025 Tampere University<br>
// MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
// MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
// MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))


// COMMAND ----------

// imports for the entire notebook
import org.apache.spark.sql.DataFrame

// COMMAND ----------

// MAGIC %md
// MAGIC ## Case 1 - Predicting the hour of the day
// MAGIC
// MAGIC This advanced task involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the ProCem research project is used as the training and test data. 
// MAGIC
// MAGIC
// MAGIC
// MAGIC The dataset is given in Parquet format, and it contains data from a period of 6 months, from May 2025 to October 2025.<br>
// MAGIC Each row contains the average of the measured values for a single minute. The following columns are included in the data:
// MAGIC
// MAGIC | column name        | column type   | description |
// MAGIC | ------------------ | ------------- | ----------- |
// MAGIC | timestamp          | timestamp     | The timestamp for this row's measurements |
// MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
// MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
// MAGIC | power_water_cooling_01 | double    | The electricity power consumed by the first water cooling machine on Kampusareena (`W`) |
// MAGIC | power_water_cooling_02 | double    | The electricity power consumed by the second water cooling machine on Kampusareena (`W`) |
// MAGIC | power_ventilation  | double        | The electricity power consumed by the ventilation machinery on Kampusareena (`W`) |
// MAGIC | power_elevator_01  | double        | The electricity power consumed by the first elevator on Kampusareena (`W`) |
// MAGIC | power_elevator_02  | double        | The electricity power consumed by the second elevator on Kampusareena (`W`) |
// MAGIC | power_ev_charging  | double        | The electricity power consumed by the electric vehicle charging station on Kampusareena (`W`) |
// MAGIC | power_solar_plant  | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
// MAGIC
// MAGIC #### Case 1 - Predicting hour of the day
// MAGIC
// MAGIC - Train a model to predict the **hour of the day** based on `temperature`, `humidity`, the `total power consumption`, and the `power produced by the solar panels`.
// MAGIC     - the total power consumption is the sum of the all six power consumption values
// MAGIC - Evaluate the accuracy of the trained model by calculating the accuracy percentage, i.e., how often it predicts the correct value, and by calculating the average hour difference between the predicted and actual hour of the day
// MAGIC     - For the accuracy measurement, you can use the Spark built-in multi-class classification evaluator, or calculate it by yourself using the prediction data frame.
// MAGIC     - For the average hour difference, consider the cyclic nature of the hour of day, i.e., for this case the difference between hours 22 and 3 is 5, the same difference as there would be between hours 17 and 22.

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import spark.implicits._

// COMMAND ----------


val kampusareenaDF = spark.read.parquet(
  "abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/kampusareena/kampusareena_measurements.parquet"
)
display(kampusareenaDF)


// COMMAND ----------

val featureCols = Array(
  "temperature", "humidity", "total_power_consumption", "power_solar_plant"
)

val case1DF = kampusareenaDF
  .withColumn("hour", hour(col("timestamp")))
  .withColumn(
    "total_power_consumption",
    col("power_water_cooling_01") +
    col("power_water_cooling_02") +
    col("power_ventilation") +
    col("power_elevator_01") +
    col("power_elevator_02") +
    col("power_ev_charging")
  )
  .na.drop(cols = featureCols :+ "hour")

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")
val assembledDF = assembler.transform(case1DF)

val Array(trainDF, testDF) = assembledDF.randomSplit(Array(0.8, 0.2), seed = 123)

// Train Random Forest
val rf_case1 = new RandomForestClassifier()
  .setLabelCol("hour")
  .setFeaturesCol("features")
  .setNumTrees(30)
val model_case1 = rf_case1.fit(trainDF)

// Predict
val predictions_case1 = model_case1.transform(testDF)

val evaluator_case1 = new MulticlassClassificationEvaluator()
  .setLabelCol("hour")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val case1Accuracy: Double = evaluator_case1.evaluate(predictions_case1)

def cyclicHourDiff(pred: Double, actual: Double): Double = {
  val diff = Math.abs(pred - actual)
  Math.min(diff, 24 - diff)
}
val avgHourDiff = predictions_case1
  .select(col("prediction"), col("hour"))
  .as[(Double, Double)]
  .map { case (pred, actual) => cyclicHourDiff(pred, actual) }
  .agg(avg("value"))
  .first.getDouble(0)

val case1AvgHourDiff: Double = avgHourDiff

// COMMAND ----------

println(s"The overall accuracy of the hour prediction model: ${scala.math.round(case1Accuracy*10000)/100.0} %")
println(f"The average hour difference between the predicted and actual hour of the day: $case1AvgHourDiff%.2f")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Case 2 - Predicting whether it is a weekend or not
// MAGIC
// MAGIC - Train a model to predict whether it is a **weekend** (Saturday or Sunday) or a weekday (Monday-Friday) based on five power values:
// MAGIC     - the total water cooling machine power consumption, i.e., the sum of the power consumptions values for the two water cooling machines
// MAGIC     - the ventilation machine power consumption
// MAGIC     - the total elevator power consumption, i.e., the sum of the power consumption values for the two elevators
// MAGIC     - the electric vehicle charging station power consumption
// MAGIC     - the power production value for the solar panels
// MAGIC - Evaluate the accuracy of the trained model by calculating the accuracy percentage, i.e., how often it predicts the correct value, and by calculating how accurate the prediction is for each separate day of the week.
// MAGIC     - For the accuracy measurement, you can use the Spark built-in multi-class classification evaluator, or calculate it by yourself using the prediction data frame.
// MAGIC     - For the separate day of the week accuracy, calculate the accuracy for predictions where the actual day was Monday, and the same for Tuesday, ...

// COMMAND ----------

val case2DF = kampusareenaDF
  .withColumn("total_water_cooling",
    col("power_water_cooling_01") + col("power_water_cooling_02")
  )
  .withColumn("total_elevator",
    col("power_elevator_01") + col("power_elevator_02")
  )
  .withColumn("weekday", dayofweek(col("timestamp")))
  .withColumn(
    "weekday_name",
    when(col("weekday") === 1, "Sunday")
      .when(col("weekday") === 2, "Monday")
      .when(col("weekday") === 3, "Tuesday")
      .when(col("weekday") === 4, "Wednesday")
      .when(col("weekday") === 5, "Thursday")
      .when(col("weekday") === 6, "Friday")
      .when(col("weekday") === 7, "Saturday")
  )
  .withColumn(
    "weekday_order",
    when(col("weekday") === 2, 1)    // Monday
      .when(col("weekday") === 3, 2) // Tuesday
      .when(col("weekday") === 4, 3) // Wednesday
      .when(col("weekday") === 5, 4) // Thursday
      .when(col("weekday") === 6, 5) // Friday
      .when(col("weekday") === 7, 6) // Saturday
      .when(col("weekday") === 1, 7) // Sunday (last)
  )
  .withColumn(
    "is_weekend",
    when(col("weekday") === 1 || col("weekday") === 7, 1).otherwise(0)
  )
  .na.drop(cols = Array(
    "total_water_cooling",
    "power_ventilation",
    "total_elevator",
    "power_ev_charging",
    "power_solar_plant",
    "is_weekend",
    "weekday"
  ))


val featureCols2 = Array(
  "total_water_cooling", "power_ventilation", "total_elevator",
  "power_ev_charging", "power_solar_plant"
)

val assembler2 = new VectorAssembler()
  .setInputCols(featureCols2)
  .setOutputCol("features")
val assembledDF2 = assembler2.transform(case2DF)

val Array(trainDF2, testDF2) = assembledDF2.randomSplit(Array(0.8, 0.2), seed = 123)

// Train Random Forest
val rf_case2 = new RandomForestClassifier()
  .setLabelCol("is_weekend")
  .setFeaturesCol("features")
  .setNumTrees(30)
val model_case2 = rf_case2.fit(trainDF2)

// Predict
val predictions_case2 = model_case2.transform(testDF2)

// Overall accuracy
val evaluator_case2 = new MulticlassClassificationEvaluator()
  .setLabelCol("is_weekend")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val case2Accuracy: Double = evaluator_case2.evaluate(predictions_case2)

val case2AccuracyDF = predictions_case2
  .groupBy("weekday", "weekday_name", "weekday_order")
  .agg(
    (sum(when(col("is_weekend") === col("prediction"), 1).otherwise(0)) / count("*") * 100)
      .alias("accuracy")
  )
  .select(
    col("weekday_name").alias("weekday"),
    round(col("accuracy"), 2).alias("accuracy")
  )
  .orderBy("weekday_order")



// COMMAND ----------

println(s"The overall accuracy of the weekend prediction model is ${scala.math.round(case2Accuracy*10000)/100.0} %")
println("Accuracy (in percentages) of the weekend predictions based on the day of the week:")
case2AccuracyDF.show(false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Case 3 - Predicting the device type
// MAGIC
// MAGIC - Train a model to predict the **device type** based on the two weather values, `temperature` and `humidity`, two timestamp related values, the `hour` and the `month`, and a `power value` from a device.
// MAGIC     - the power values should be divided into five categories, i.e., device types:
// MAGIC         - `elevator`: for the sum of the power consumption values for the two elevators
// MAGIC         - `ev_charging`: for the electric vehicle charging station power consumption
// MAGIC         - `solar_plant`: the power production value for the solar panels
// MAGIC         - `ventilation`: the ventilation machine power consumption
// MAGIC         - `water_cooling`: the sum of the power consumptions values for the two water cooling machines
// MAGIC - Evaluate the accuracy of the trained model by calculating the accuracy percentage, i.e., how often it predicts the correct value, and by calculating how accurate the prediction is for each separate device type.
// MAGIC     - For the accuracy measurement, you can use the Spark built-in multi-class classification evaluator, or calculate it by yourself using the prediction data frame.
// MAGIC     - For the separate device type accuracy, calculate the accuracy for predictions where the power values were for elevators, and the same for the ventilation, ...

// COMMAND ----------

// Add hour and month columns
val baseDF = kampusareenaDF
  .withColumn("hour", hour(col("timestamp")))
  .withColumn("month", month(col("timestamp")))

// Create a long-format DataFrame for each device type
val elevatorDF = baseDF
  .withColumn("device_type", lit("elevator"))
  .withColumn("power_value", col("power_elevator_01") + col("power_elevator_02"))
  .select("temperature", "humidity", "hour", "month", "power_value", "device_type")

val evChargingDF = baseDF
  .withColumn("device_type", lit("ev_charging"))
  .withColumn("power_value", col("power_ev_charging"))
  .select("temperature", "humidity", "hour", "month", "power_value", "device_type")

val solarPlantDF = baseDF
  .withColumn("device_type", lit("solar_plant"))
  .withColumn("power_value", col("power_solar_plant"))
  .select("temperature", "humidity", "hour", "month", "power_value", "device_type")

val ventilationDF = baseDF
  .withColumn("device_type", lit("ventilation"))
  .withColumn("power_value", col("power_ventilation"))
  .select("temperature", "humidity", "hour", "month", "power_value", "device_type")

val waterCoolingDF = baseDF
  .withColumn("device_type", lit("water_cooling"))
  .withColumn("power_value", col("power_water_cooling_01") + col("power_water_cooling_02"))
  .select("temperature", "humidity", "hour", "month", "power_value", "device_type")

// Union all device DataFrames
val case3DF = elevatorDF
  .union(evChargingDF)
  .union(solarPlantDF)
  .union(ventilationDF)
  .union(waterCoolingDF)

// Remove rows with nulls in any feature or label column
val case3DFClean = case3DF.na.drop(cols = Array(
  "temperature", "humidity", "hour", "month", "power_value", "device_type"
))

// COMMAND ----------

// Index the label column because the label column needs to be numeric
val indexer = new StringIndexer()
  .setInputCol("device_type")
  .setOutputCol("label")
val indexedDF = indexer.fit(case3DFClean).transform(case3DFClean)

// Assemble features
val featureCols3 = Array("temperature", "humidity", "hour", "month", "power_value")
val assembler3 = new VectorAssembler()
  .setInputCols(featureCols3)
  .setOutputCol("features")
val assembledDF3 = assembler3.transform(indexedDF)

// Split data
val Array(trainDF3, testDF3) = assembledDF3.randomSplit(Array(0.8, 0.2), seed = 123)

// Train Random Forest
val rf3 = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setNumTrees(30)
val model3 = rf3.fit(trainDF3)

// COMMAND ----------

// Predict on test data
val predictions3 = model3.transform(testDF3)

// Overall accuracy using Spark's evaluator
val evaluator3 = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val case3Accuracy: Double = evaluator3.evaluate(predictions3)

val labelToDeviceType = indexedDF
  .select("device_type", "label")
  .distinct()
  .collect()
  .map(row => (row.getAs[Double]("label"), row.getAs[String]("device_type")))
  .toMap

// Add a column with the original device_type for easier grouping
val predictionsWithType = predictions3.withColumn(
  "device_type",
  expr(s"CASE " + labelToDeviceType.map { case (idx, name) => s"WHEN label = $idx THEN '$name'" }.mkString(" ") + " END")
)

// Calculate accuracy for each device_type
val case3AccuracyDF = predictionsWithType
  .groupBy("device_type")
  .agg(
      round((sum(when(col("label") === col("prediction"), 1).otherwise(0)) / count("*") * 100), 2).alias("accuracy")
  )
  .select("device_type", "accuracy")
  .orderBy(col("accuracy").desc)

// COMMAND ----------

println(s"The overall accuracy of the device type prediction model is ${scala.math.round(case3Accuracy*10000)/100.0} %")
println("Accuracy (in percentages) of the device predictions based on the device:")
case3AccuracyDF.show(false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Case 4 
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC This case uses a Logistic Regression model that predicts whether a measurement is taken in a month that has 30 days or 31 days (it is a binary-classification problem). The features columns (taken from kampusareenaDF) will be: temperature, humidity and two power values (out of the 5 power values total_water_cooling, ventilation, total_elevator, power_ev_charging, and power_solar_plant) with the largest standard deviations.

// COMMAND ----------

// Add month column
val kampusareenaDF_With_Month = kampusareenaDF.withColumn("month", month(col("timestamp")))

// Map month to binary label: 1 if 31 days, 0 if 30 days (April, June, September have 30 days; May, July, August, October have 31)
val kampusareenaDF_WithLabel = kampusareenaDF_With_Month.withColumn(
  "label",
  when(col("month").isin(4, 6, 9), 0).otherwise(1)
)

// Calculate stddev for each power value
val kampusareenaWithTotals = kampusareenaDF_WithLabel
  .withColumn("total_water_cooling", col("power_water_cooling_01") + col("power_water_cooling_02"))
  .withColumn("total_elevator", col("power_elevator_01") + col("power_elevator_02"))

// Compute stddev for each power column
val stds = Seq(
  "total_water_cooling",
  "power_ventilation",
  "total_elevator",
  "power_ev_charging",
  "power_solar_plant"
).map(c => (c, kampusareenaWithTotals.agg(stddev_pop(col(c))).first().getDouble(0)))

// Select two power columns with largest stddev
val top2PowerCols = stds.sortBy(-_._2).take(2).map(_._1)

// Select relevant columns and drop rows with nulls
val case4_finalDF = kampusareenaWithTotals
  .select(
    col("temperature"),
    col("humidity"),
    col(top2PowerCols(0)),
    col(top2PowerCols(1)),
    col("label")
  )
  .na.drop()

//display(case4_finalDF)

// COMMAND ----------

// Assemble features
val assembler4 = new VectorAssembler()
  .setInputCols(Array("temperature", "humidity", top2PowerCols(0), top2PowerCols(1)))
  .setOutputCol("features")
val assembledDF4 = assembler4.transform(case4_finalDF)

// Split data
val Array(trainDF4, testDF4) = assembledDF4.randomSplit(Array(0.8, 0.2), seed = 42)

// Train logistic regression model
val lr = new LogisticRegression()
  .setLabelCol("label")
  .setFeaturesCol("features") // default threshold is 0.5
val model4 = lr.fit(trainDF4)

val predictions4 = model4.transform(testDF4)

val evaluator4 = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")

// Accuracy
val accuracy4 = evaluator4.setMetricName("accuracy").evaluate(predictions4)
// Weighted Precision
val weightedPrecision4 = evaluator4.setMetricName("weightedPrecision").evaluate(predictions4)

// COMMAND ----------

println(s"Accuracy of Case 4 model is: ${scala.math.round(accuracy4*10000)/100.0} %")
println(s"Weighted Precision of Case 4 model is: ${scala.math.round(weightedPrecision4*100)/100.0}")