# Airport flight prediction using three simple PySpark ML models
# Includes basic data analysis before training the models.

from datetime import timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Change these paths if your files are somewhere else
INPUT_PATH = "/content/airport_traffic_2025.csv"
OUTPUT_PATH = "/content/airport_results"

spark = SparkSession.builder.appName("AirportTrafficModels").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

try:
    # 1. Read the CSV
    raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(INPUT_PATH)
    )

    # ==================== DATA ANALYSIS ====================
    print("\n========== 1. BASIC DATA INFORMATION ==========")
    print("Number of rows:", raw.count())
    print("Number of columns:", len(raw.columns))
    print("Column names:", raw.columns)

    print("\nColumn data types:")
    raw.printSchema()

    print("\nFirst 10 rows:")
    raw.show(10, truncate=False)

    print("\n========== 2. STATISTICAL SUMMARY ==========")
    numeric_cols = [
        field.name for field in raw.schema.fields
        if isinstance(field.dataType, NumericType)
    ]

    if numeric_cols:
        raw.select(*numeric_cols).describe().show(truncate=False)
    else:
        print("No numeric columns were detected.")

    print("\n========== 3. MISSING VALUES ==========")
    raw.select([
        F.sum(
            F.when(
                F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == ""),
                1
            ).otherwise(0)
        ).alias(c)
        for c in raw.columns
    ]).show(truncate=False)

    print("\n========== 4. DUPLICATE ROWS ==========")
    print("Duplicate full rows:", raw.count() - raw.dropDuplicates().count())

    print("\n========== 5. DATASET COUNTS ==========")
    raw.agg(
        F.countDistinct("APT_ICAO").alias("unique_airports"),
        F.countDistinct("FLT_DATE").alias("unique_dates"),
        F.countDistinct("STATE_NAME").alias("unique_states_or_countries")
    ).show()

    print("\n========== 6. MONTHLY FLIGHT TOTALS ==========")
    raw.withColumn(
        "month", F.month(F.to_date(F.col("FLT_DATE").cast("string")))
    ).groupBy("month").agg(
        F.sum(F.col("FLT_TOT_1").cast("double")).alias("total_flights")
    ).orderBy("month").show(12)

    print("\n========== 7. TEN BUSIEST AIRPORTS ==========")
    raw.groupBy("APT_ICAO", "APT_NAME").agg(
        F.sum(F.col("FLT_TOT_1").cast("double")).alias("total_flights"),
        F.round(F.avg(F.col("FLT_TOT_1").cast("double")), 2).alias("average_flights")
    ).orderBy(F.desc("total_flights")).show(10, truncate=False)

    # ==================== PREPARE DATA FOR ML ====================
    print("\n========== 8. PREPARE DATA FOR MACHINE LEARNING ==========")
    df = raw.select(
        F.to_date(F.col("FLT_DATE").cast("string")).alias("date"),
        F.trim(F.col("APT_ICAO").cast("string")).alias("APT_ICAO"),
        F.trim(F.col("APT_NAME").cast("string")).alias("APT_NAME"),
        F.col("FLT_TOT_1").cast("double").alias("flights")
    )

    df = df.dropna().filter(F.col("flights") >= 0)
    df = df.dropDuplicates(["date", "APT_ICAO"])

    print("Clean data sample:")
    df.show(5, truncate=False)
    print("Rows after cleaning:", df.count())

    # 9. Add the flight count from yesterday and from 7 days ago
    yesterday = df.select(
        "APT_ICAO",
        F.date_add("date", 1).alias("date"),
        F.col("flights").alias("yesterday")
    )

    last_week = df.select(
        "APT_ICAO",
        F.date_add("date", 7).alias("date"),
        F.col("flights").alias("last_week")
    )

    data = df.join(yesterday, ["APT_ICAO", "date"], "left")
    data = data.join(last_week, ["APT_ICAO", "date"], "left")
    data = data.dropna(subset=["yesterday", "last_week"])

    print("Rows with both previous-day and previous-week data:", data.count())

    # 10. Train on earlier dates and test on later dates
    # This cutoff assumes the CSV contains data from 2025.
    train = data.filter(F.col("date") < F.to_date(F.lit("2025-10-01")))
    test = data.filter(F.col("date") >= F.to_date(F.lit("2025-10-01")))

    if train.limit(1).count() == 0 or test.limit(1).count() == 0:
        raise ValueError(
            "Training or test data is empty. Check the dates and the 2025-10-01 cutoff."
        )

    print("Training rows:", train.count())
    print("Testing rows:", test.count())

    # 11. Put the two input columns into the format Spark ML needs
    assembler = VectorAssembler(
        inputCols=["yesterday", "last_week"],
        outputCol="features"
    )

    train = assembler.transform(train)
    test = assembler.transform(test)

    # 12. Compare a simple baseline with three machine-learning models
    # The baseline just assumes today's flights will equal yesterday's.
    mae_evaluator = RegressionEvaluator(
        labelCol="flights", predictionCol="prediction", metricName="mae"
    )
    rmse_evaluator = RegressionEvaluator(
        labelCol="flights", predictionCol="prediction", metricName="rmse"
    )

    baseline_rmse = RegressionEvaluator(
        labelCol="flights", predictionCol="yesterday", metricName="rmse"
    ).evaluate(test)
    baseline_mae = RegressionEvaluator(
        labelCol="flights", predictionCol="yesterday", metricName="mae"
    ).evaluate(test)

    results = [("Previous-day baseline", baseline_mae, baseline_rmse)]

    models = [
        ("Linear Regression", LinearRegression(
            featuresCol="features", labelCol="flights"
        )),
        ("Random Forest", RandomForestRegressor(
            featuresCol="features", labelCol="flights",
            numTrees=20, maxDepth=5, seed=42
        )),
        ("Gradient-Boosted Trees", GBTRegressor(
            featuresCol="features", labelCol="flights",
            maxIter=20, maxDepth=3, seed=42
        ))
    ]

    best_model = None
    best_name = ""
    best_rmse = None
    best_predictions = None

    for name, algorithm in models:
        print("Training", name, "...")
        fitted_model = algorithm.fit(train)
        predictions = fitted_model.transform(test)

        # Flight counts cannot be negative, so set negative predictions to zero.
        predictions = predictions.withColumn(
            "prediction", F.greatest(F.col("prediction"), F.lit(0.0))
        )

        mae = mae_evaluator.evaluate(predictions)
        rmse = rmse_evaluator.evaluate(predictions)
        results.append((name, mae, rmse))
        print(name, "MAE:", round(mae, 2), "RMSE:", round(rmse, 2))

        # Choose the model with the lowest test RMSE.
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_model = fitted_model
            best_name = name
            best_predictions = predictions

    print("\nModel comparison (lower MAE and RMSE are better):")
    results.sort(key=lambda row: row[2])
    spark.createDataFrame(results, ["model", "MAE", "RMSE"]).show(truncate=False)
    print("Best machine-learning model by test RMSE:", best_name)

    print("Actual and predicted flights using the selected model:")
    best_predictions.select(
        "date", "APT_ICAO", "APT_NAME", "flights", "prediction"
    ).orderBy("date").show(20, truncate=False)

    # 13. Predict flights for the day after the last date in the CSV
    last_date = df.agg(F.max("date")).first()[0]
    if last_date is None:
        raise ValueError("No valid dates were found in the CSV.")

    forecast_date = last_date + timedelta(days=1)
    print("Forecast date:", forecast_date)

    # For the forecast date, yesterday's flights are the flights on the last date.
    forecast = df.filter(F.col("date") == F.lit(last_date)).select(
        "APT_ICAO", "APT_NAME", F.col("flights").alias("yesterday")
    ).withColumn("date", F.lit(forecast_date).cast("date"))

    # Join the value from seven days before the forecast date.
    forecast = forecast.join(last_week, ["APT_ICAO", "date"], "left")
    forecast = forecast.dropna(subset=["yesterday", "last_week"])
    forecast = assembler.transform(forecast)

    forecast = best_model.transform(forecast).withColumn(
        "prediction", F.greatest(F.col("prediction"), F.lit(0.0))
    )

    print("Next-day predictions using", best_name, ":")
    forecast.select(
        "date", "APT_ICAO", "APT_NAME", "yesterday", "last_week", "prediction"
    ).orderBy(F.desc("prediction")).show(20, truncate=False)

    # 14. Save the test predictions and next-day forecast
    best_predictions.select(
        "date", "APT_ICAO", "APT_NAME", "flights", "prediction"
    ).coalesce(1).write.mode("overwrite").option("header", True).csv(
        OUTPUT_PATH + "/test_predictions"
    )

    forecast.select(
        "date", "APT_ICAO", "APT_NAME", "yesterday", "last_week", "prediction"
    ).coalesce(1).write.mode("overwrite").option("header", True).csv(
        OUTPUT_PATH + "/next_day_predictions"
    )

    print("Results saved in:", OUTPUT_PATH)

finally:
    spark.stop()
