from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("ADF_Databricks_Pipeline").getOrCreate()

# Read from Azure Data Lake (replace path with actual)
df = spark.read.option("header", True).csv("abfss://your-container@your-account.dfs.core.windows.net/sensor-data.csv")

# Clean data
df_clean = df.withColumn("temperature", col("temperature").cast("float")) \
             .withColumn("torque", col("torque").cast("float")) \
             .withColumn("vibration", col("vibration").cast("float")) \
             .dropna()

# Save the cleaned data
df_clean.write.mode("overwrite").parquet("abfss://your-container@your-account.dfs.core.windows.net/processed-data/")
