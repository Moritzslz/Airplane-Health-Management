import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# TODO dont forget to stop the SparkSession with spark.stop()

pd_df = pd.read_csv("data/sample_sensor_data.csv")

print(pd_df.head())

spark = SparkSession.builder.appName("AirplaneHealthMonitoring").getOrCreate()

df = spark.read.csv("data/sample_sensor_data.csv", header=True, inferSchema=True)

df.show()

df.printSchema()

df.describe().show()

temperature_df = df.filter(df["reading_type"] == "Temperature").toPandas()
temperature_df.plot(x="timestamp", y="value", kind="line")
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.title('Temperature Readings Over Time')
plt.show()


def get_max(spark_df, reading_type, return_pandas_df):
    max_value = spark_df.filter(spark_df["reading_type"] == reading_type).agg({"value": "max"}).collect()[0][0]
    print(reading_type + "max value:", max_value)
    if return_pandas_df:
        return max_value.toPandas()
    return max_value


def get_min(spark_df, reading_type, return_pandas_df):
    min_value = spark_df.filter(spark_df["reading_type"] == reading_type).agg({"value": "min"}).collect()[0][0]
    print(reading_type + "min value:", min_value)
    if return_pandas_df:
        return min_value.toPandas()
    return min_value


def get_avg(spark_df, reading_type, return_pandas_df):
    avg_value = spark_df.filter(spark_df["reading_type"] == reading_type).agg({"value": "avg"}).collect()[0][0]
    print(reading_type + "avg value:", avg_value)
    if return_pandas_df:
        return avg_value.toPandas()
    return avg_value


def plot_data(title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

spark.stop()