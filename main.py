import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# TODO dont forget to stop the SparkSession with spark.stop()
# Pandas:
# Pandas DataFrame: Print first rows with .head() and last rows with .tail()
# df.shape = number of rows and number of columns
# shape is an attribute (array) of the df
# df.info gives total number of rows, columns, datatypes and memory requirement
# df.isnull().sum() to check for null values

pd_df = pd.read_csv("data/flight_01_data.csv")

spark = SparkSession.builder.appName("AirplaneHealthMonitoring").getOrCreate()

spark_df = spark.read.csv("data/flight_01_data.csv", header=True, inferSchema=True)
spark_df.printSchema()
spark_df.describe().show()

def create_plot_for_all_engines(number_of_engines):

    all_voltage_readings_by_engine = []
    all_rpm_readings_by_engine = []
    all_thrust_readings_by_engine = []
    all_temperature_readings_by_engine = []

    def create_plot_for_individual_engine(engine_id):
            engine_df = spark_df.filter(spark_df["engine_id"] == engine_id).toPandas()

            timestamps = engine_df[engine_df["reading_type"] == "voltage"]["timestamp"]
            voltage_readings = engine_df[engine_df["reading_type"] == "voltage"]["value"]
            rpm_readings = engine_df[engine_df["reading_type"] == "rotor_speed_rpm"]["value"]
            thrust_readings = engine_df[engine_df["reading_type"] == "thrust_newton"]["value"]
            temperature_readings = engine_df[engine_df["reading_type"] == "temperature_celsius"]["value"]

            all_voltage_readings_by_engine.append((timestamps, voltage_readings))
            all_rpm_readings_by_engine.append((timestamps, rpm_readings))
            all_thrust_readings_by_engine.append((timestamps, thrust_readings))
            all_temperature_readings_by_engine.append((timestamps, temperature_readings))

            figure, axis = plt.subplots()
            axis.plot(timestamps, voltage_readings, label="Voltage")
            axis.plot(timestamps, rpm_readings, label="Rotor speed rpm")
            axis.plot(timestamps, thrust_readings, label="Thrust in newton")
            axis.plot(timestamps, temperature_readings, label="Temperature in celsius")
            axis.set_title("Readings of " + engine_id)
            axis.set_xlabel("Time")
            axis.legend()

            plt.show()

    for n in range(1, number_of_engines + 1):
        engine_id = "engine_" + str(n)
        create_plot_for_individual_engine(engine_id)

        # Plot for all engines' voltage readings
    for index, (timestamps, voltage_readings) in enumerate(all_voltage_readings_by_engine):
        plt.plot(timestamps, voltage_readings, label=index + 1)
    plt.title("Voltage Readings of All Engines")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.legend()
    plt.show()

    # Plot for all engines' RPM readings
    for index, (timestamps, rpm_readings) in enumerate(all_rpm_readings_by_engine):
        plt.plot(timestamps, rpm_readings, label=index + 1)
    plt.title("RPM Readings of All Engines")
    plt.xlabel("Time")
    plt.ylabel("Rotor Speed (RPM)")
    plt.legend()
    plt.show()

    # Plot for all engines' thrust readings
    for index, (timestamps, thrust_readings) in enumerate(all_thrust_readings_by_engine):
        plt.plot(timestamps, thrust_readings, label=index + 1)
    plt.title("Thrust Readings of All Engines")
    plt.xlabel("Time")
    plt.ylabel("Thrust (Newton)")
    plt.legend()
    plt.show()

    # Plot for all engines' temperature readings
    for index, (timestamps, temperature_readings) in enumerate(all_temperature_readings_by_engine):
        plt.plot(timestamps, temperature_readings, label=index + 1)
    plt.title("Temperature Readings of All Engines")
    plt.xlabel("Time")
    plt.ylabel("Temperature (Celsius)")
    plt.legend()
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


create_plot_for_all_engines(3)

spark.stop()
