import os
from datetime import timedelta
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Raw string prefix (r"...") prevents Python from interpreting \U as a Unicode escape sequence
INPUT_PATH = r"C:\Users\yashm\Desktop\spark_Data\airport_traffic_2025.csv"
OUTPUT_PATH = r"C:\Users\yashm\Desktop\spark_Data\airport_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)

spark = SparkSession.builder.appName("AirportTrafficModels").getOrCreate()

raw = spark.read.option("header", True).option("inferSchema", True).csv(INPUT_PATH)

print("DATASET INFORMATION")
print("Rows:", raw.count())
print("Columns:", len(raw.columns))
print("Column names:", raw.columns)
raw.printSchema()
raw.show(10, truncate=False)

print("STATISTICAL SUMMARY")
raw.describe("FLT_TOT_1").show()

print("MISSING VALUES")
raw.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in raw.columns
]).show(truncate=False)

print("DUPLICATE ROWS:", raw.count() - raw.dropDuplicates().count())

print("DATASET COUNTS")
raw.agg(
    F.countDistinct("APT_ICAO").alias("airports"),
    F.countDistinct("FLT_DATE").alias("dates"),
    F.countDistinct("STATE_NAME").alias("states_or_countries")
).show()

print("MONTHLY FLIGHT TOTALS")
raw.groupBy(
    F.month(F.to_date(F.col("FLT_DATE").cast("string"))).alias("month")
).agg(
    F.sum(F.col("FLT_TOT_1").cast("double")).alias("total_flights")
).orderBy("month").show()

print("10 BUSIEST AIRPORTS")
raw.groupBy("APT_ICAO", "APT_NAME").agg(
    F.sum(F.col("FLT_TOT_1").cast("double")).alias("total_flights"),
    F.round(F.avg(F.col("FLT_TOT_1").cast("double")), 2).alias("average_flights")
).orderBy(F.desc("total_flights")).show(10, truncate=False)

df = raw.select(
    F.to_date(F.col("FLT_DATE").cast("string")).alias("date"),
    F.trim(F.col("APT_ICAO").cast("string")).alias("APT_ICAO"),
    F.trim(F.col("APT_NAME").cast("string")).alias("APT_NAME"),
    F.col("FLT_TOT_1").cast("double").alias("flights")
)

df = df.dropna().filter(F.col("flights") >= 0)
df = df.dropDuplicates(["date", "APT_ICAO"])

print("CLEAN DATA")
df.show(5, truncate=False)
print("Rows after cleaning:", df.count())

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

train = data.filter(F.col("date") < F.lit("2025-10-01").cast("date"))
test = data.filter(F.col("date") >= F.lit("2025-10-01").cast("date"))

train_count = train.count()
test_count = test.count()

print("Training rows:", train_count)
print("Testing rows:", test_count)

if train_count == 0 or test_count == 0:
    raise ValueError("Training or testing data is empty. Check the date cutoff.")

assembler = VectorAssembler(
    inputCols=["yesterday", "last_week"],
    outputCol="features"
)

train = assembler.transform(train)
test = assembler.transform(test)

mae_evaluator = RegressionEvaluator(
    labelCol="flights",
    predictionCol="prediction",
    metricName="mae"
)

rmse_evaluator = RegressionEvaluator(
    labelCol="flights",
    predictionCol="prediction",
    metricName="rmse"
)

baseline_mae = RegressionEvaluator(
    labelCol="flights",
    predictionCol="yesterday",
    metricName="mae"
).evaluate(test)

baseline_rmse = RegressionEvaluator(
    labelCol="flights",
    predictionCol="yesterday",
    metricName="rmse"
).evaluate(test)

results = [("Previous-day baseline", baseline_mae, baseline_rmse)]

models = [
    ("Linear Regression", LinearRegression(
        featuresCol="features",
        labelCol="flights"
    )),
    ("Random Forest", RandomForestRegressor(
        featuresCol="features",
        labelCol="flights",
        numTrees=20,
        maxDepth=5,
        seed=42
    )),
    ("Gradient-Boosted Trees", GBTRegressor(
        featuresCol="features",
        labelCol="flights",
        maxIter=20,
        maxDepth=3,
        seed=42
    ))
]

best_model = None
best_name = ""
best_rmse = float("inf")
best_predictions = None

for name, algorithm in models:
    print("Training:", name)

    model = algorithm.fit(train)
    predictions = model.transform(test)

    predictions = predictions.withColumn(
        "prediction",
        F.greatest(F.col("prediction"), F.lit(0.0))
    )

    mae = mae_evaluator.evaluate(predictions)
    rmse = rmse_evaluator.evaluate(predictions)

    results.append((name, mae, rmse))

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    if rmse < best_rmse:
        best_model = model
        best_name = name
        best_rmse = rmse
        best_predictions = predictions

print("MODEL COMPARISON")

comparison = spark.createDataFrame(
    results, ["model", "MAE", "RMSE"]
).orderBy("RMSE")

comparison.show(truncate=False)

print("Best model:", best_name)

print("ACTUAL VS PREDICTED FLIGHTS")
best_predictions.select(
    "date", "APT_ICAO", "APT_NAME", "flights", "prediction"
).orderBy("date").show(20, truncate=False)

last_date = df.agg(F.max("date")).first()[0]

if last_date is None:
    raise ValueError("No valid dates found in the dataset.")

forecast_date = last_date + timedelta(days=1)

forecast = df.filter(F.col("date") == F.lit(last_date)).select(
    "APT_ICAO",
    "APT_NAME",
    F.col("flights").alias("yesterday")
)

forecast = forecast.withColumn("date", F.lit(forecast_date).cast("date"))
forecast = forecast.join(last_week, ["APT_ICAO", "date"], "left")
forecast = forecast.dropna(subset=["yesterday", "last_week"])
forecast = assembler.transform(forecast)

forecast = best_model.transform(forecast).withColumn(
    "prediction",
    F.greatest(F.col("prediction"), F.lit(0.0))
)

print("FORECAST DATE:", forecast_date)

forecast.select(
    "date", "APT_ICAO", "APT_NAME", "yesterday", "last_week", "prediction"
).orderBy(F.desc("prediction")).show(20, truncate=False)

best_predictions.select(
    "date", "APT_ICAO", "APT_NAME", "flights", "prediction"
).coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(OUTPUT_PATH + "/test_predictions")

forecast.select(
    "date", "APT_ICAO", "APT_NAME", "yesterday", "last_week", "prediction"
).coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(OUTPUT_PATH + "/next_day_predictions")

print("Results saved to:", OUTPUT_PATH)

monthly = raw.withColumn(
    "month",
    F.month(F.to_date(F.col("FLT_DATE").cast("string")))
).groupBy("month").agg(
    F.sum(F.col("FLT_TOT_1").cast("double")).alias("total_flights")
).orderBy("month").toPandas()

month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

monthly["month_name"] = monthly["month"].map(month_names)

plt.figure(figsize=(10, 5))
plt.bar(monthly["month_name"], monthly["total_flights"])
plt.title("Monthly Flight Traffic")
plt.xlabel("Month")
plt.ylabel("Total Flights")
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "/monthly_flights.png", dpi=150)
plt.show()
plt.close()

top_airports = raw.groupBy("APT_NAME").agg(
    F.sum(F.col("FLT_TOT_1").cast("double")).alias("total_flights")
).orderBy(F.desc("total_flights")).limit(10).toPandas()

top_airports = top_airports.sort_values("total_flights")

plt.figure(figsize=(10, 6))
plt.barh(top_airports["APT_NAME"], top_airports["total_flights"])
plt.title("Top 10 Busiest Airports")
plt.xlabel("Total Flights")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "/busiest_airports.png", dpi=150)
plt.show()
plt.close()

comparison_pd = comparison.toPandas()

comparison_pd.plot(
    x="model",
    y=["MAE", "RMSE"],
    kind="bar",
    figsize=(10, 5)
)
plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Prediction Error")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "/model_comparison.png", dpi=150)
plt.show()
plt.close()

daily_predictions = best_predictions.groupBy("date").agg(
    F.sum("flights").alias("actual_flights"),
    F.sum("prediction").alias("predicted_flights")
).orderBy("date").toPandas()

plt.figure(figsize=(12, 5))
plt.plot(
    daily_predictions["date"],
    daily_predictions["actual_flights"],
    label="Actual Flights"
)
plt.plot(
    daily_predictions["date"],
    daily_predictions["predicted_flights"],
    label="Predicted Flights"
)
plt.title("Actual vs Predicted Daily Flight Traffic")
plt.xlabel("Date")
plt.ylabel("Total Flights Across Airports")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "/actual_vs_predicted.png", dpi=150)
plt.show()
plt.close()

errors = best_predictions.select(
    (F.col("flights") - F.col("prediction")).alias("error")
).sample(False, 0.2, seed=42).limit(10000).toPandas()

plt.figure(figsize=(10, 5))
plt.hist(errors["error"].dropna(), bins=40, edgecolor="black")
plt.title("Prediction Error Distribution")
plt.xlabel("Actual Flights - Predicted Flights")
plt.ylabel("Number of Predictions")
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "/prediction_errors.png", dpi=150)
plt.show()
plt.close()

forecast_pd = forecast.select(
    "APT_NAME", "prediction"
).orderBy(
    F.desc("prediction")
).limit(10).toPandas()

if not forecast_pd.empty:
    forecast_pd = forecast_pd.sort_values("prediction")

    plt.figure(figsize=(10, 6))
    plt.barh(forecast_pd["APT_NAME"], forecast_pd["prediction"])
    plt.title("Top 10 Airports by Next-Day Forecast")
    plt.xlabel("Predicted Flights")
    plt.ylabel("Airport")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + "/next_day_forecast.png", dpi=150)
    plt.show()
    plt.close()

print("All graphs saved in:", OUTPUT_PATH)

spark.stop()