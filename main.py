from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# Loading the dataset
spark = SparkSession.builder.appName("AirplaneHealthMonitoring").getOrCreate()
spark_df = spark.read.csv("data/flight_01_data.csv", header=True, inferSchema=True)
spark_df.printSchema()
spark_df.describe().show()

# Example thresholds
thresholds_low = {
    "voltage": "290",
    "rotor_speed_rpm": "1000",
    "thrust_newton": "50",
    "temperature_celsius": "8"
}

thresholds_high = {
    "voltage": "350",
    "rotor_speed_rpm": "9000",
    "thrust_newton": "200",
    "temperature_celsius": "90"
}

"""
This dictionary stores all engines and their high-level (traffic light) status
Red: Readings have been out of bounds of the thresholds 
Yellow: Readings have been within 10% of the thresholds
Green: Readings ok.
"""
traffic_light = {}


def analyse_engine_data(engine_id, begin_of_analysis, visualise):
    """
        This function analyses voltage, rpm, thrust and temperature readings
        of an engine and alters if a certain threshold has been exceeded.

        Parameters:
        engine_id (String): Specify the engine you want to analise. Keep the format of the dataset in mind.
        begin_of_analysis(String): Specify from which point on the data should be analysed; Format: 2024-02-09 08:00:00
        visualise (Boolean):

        Returns:
        Console output giving high level information about the status of an engine
        Optionally: Graph of engines data
    """

    begin_datetime = datetime.strptime(begin_of_analysis, "%Y-%m-%d %H:%M:%S")
    engine_df = spark_df.filter((col("engine_id") == engine_id) & (col("timestamp") >= begin_datetime))

    # For testing in development
    # engine_df.show()

    timestamps = engine_df.select("timestamp").collect()
    voltage_readings = engine_df.where(col("reading_type") == "voltage").select("value")
    rpm_readings = engine_df.where(col("reading_type") == "rotor_speed_rpm").select("value")
    thrust_readings = engine_df.where(col("reading_type") == "thrust_newton").select("value")
    temperature_readings = engine_df.where(col("reading_type") == "temperature_celsius").select("value")

    # Check if readings are within thresholds
    check_threshold_and_alert(engine_id, voltage_readings, "voltage")
    check_threshold_and_alert(engine_id, rpm_readings, "rotor_speed_rpm")
    check_threshold_and_alert(engine_id, thrust_readings, "thrust_newton")
    check_threshold_and_alert(engine_id, temperature_readings, "temperature_celsius")

    if visualise:
        visualise_engine_data(engine_id, timestamps, voltage_readings, rpm_readings, thrust_readings, temperature_readings)


def check_threshold_and_alert(engine_id, data, reading_type):
    """
        This function compares the data to the min and max thresholds and sets the traffic light for the engine.
        An alert to the console is printed when the traffic light is set to yellow or red.

        Parameters:
        engine_id (String): Specify the engine you want to analise. Keep the format of the dataset in mind.
        data(spark_df): readings of a specific type
        reading_type (String): Exact type as specified in the thresholds dictionaries

        Returns:
        Console output giving high level information about the status of an engine

    """

    min_threshold = float(thresholds_low.get(reading_type))
    max_threshold = float(thresholds_high.get(reading_type))

    min_value = get_min(data)
    max_value = get_max(data)

    # Calculate 10% threshold values
    min_threshold_110_percent = min_threshold * 1.1
    max_threshold_90_percent = max_threshold * 0.9

    # Traffic light: green
    traffic_light[engine_id] = "green"

    # Traffic light: yellow
    if min_value <= min_threshold_110_percent:
        traffic_light[engine_id] = "yellow"
    if max_value >= max_threshold_90_percent:
        traffic_light[engine_id] = "yellow"

    # Traffic light: red
    if min_value <= min_threshold:
        alert(engine_id, reading_type, min_value, 0)
        traffic_light[engine_id] = "red"
    if max_value >= max_threshold:
        alert(engine_id, reading_type, max_value, 1)
        traffic_light[engine_id] = "red"


def visualise_engine_data(engine_id, timestamps, voltage_readings, rpm_readings, thrust_readings, temperature_readings):
    """
        This function maps the spark dataframes to pandas and creates plots for each reading.

        Parameters:
        timestamps (spark_df)
        voltage_readings (spark_df)
        rpm_readings (spark_df)
        thrust_readings (spark_df)
        temperature_readings (spark_df)

        Returns:
        Graphs
    """

    time = [row["timestamp"] for row in timestamps]
    time = list(dict.fromkeys(time))  # Removing all duplicates
    voltage_readings = voltage_readings.toPandas()
    rpm_readings = rpm_readings.toPandas()
    thrust_readings = thrust_readings.toPandas()
    temperature_readings = temperature_readings.toPandas()

    voltage = voltage_readings["value"].tolist()
    rpm = rpm_readings["value"].tolist()
    thrust = thrust_readings["value"].tolist()
    temperature = temperature_readings["value"].tolist()

    engine_id = engine_id.upper()
    plt.figure(figsize=(10, 6))

    # Plot voltage readings
    plt.subplot(2, 2, 1)
    plt.plot(time, voltage, color="blue")
    plt.title("Voltage Readings" + " : " + engine_id)
    plt.xlabel("Timestamp")
    plt.ylabel("Voltage")

    # Plot RPM readings
    plt.subplot(2, 2, 2)
    plt.plot(time, rpm, color="green")
    plt.title("Rotor Speed (RPM) Readings" + " : " + engine_id)
    plt.xlabel("Timestamp")
    plt.ylabel("Rotor Speed (RPM)")

    # Plot thrust readings
    plt.subplot(2, 2, 3)
    plt.plot(time, thrust, color="red")
    plt.title("Thrust (Newton) Readings" + " : " + engine_id)
    plt.xlabel("Timestamp")
    plt.ylabel("Thrust (Newton)")

    # Plot temperature readings
    plt.subplot(2, 2, 4)
    plt.plot(time, temperature, color="orange")
    plt.title("Temperature (Celsius) Readings" + " : " + engine_id)
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (Celsius)")

    plt.tight_layout()
    plt.show()


def alert(engine_id, reading_type, value, type_of_threshold):
    """
        Alerts to console

        Parameters:
        engine_id (String): Specify the engine you want to analise. Keep the format of the dataset in mind.
        reading_type (String):
        type (int): 0 == minimum threshold; 1 == maximum threshold

        Returns:
        Console output
    """

    print("************************************************************")
    print("Alert for " + engine_id.upper() + ":")
    if type_of_threshold == 0:
        print(reading_type.upper() + " readings LOWER than threshold: " + str(value))
    else:
        print(reading_type.upper() + " readings HIGHER than threshold: " + str(value))
    print("************************************************************")


def get_max(readings_df):
    max_value = readings_df.agg({"value": "max"}).collect()[0][0]
    return float(max_value)


def get_min(readings_df):
    min_value = readings_df.agg({"value": "min"}).collect()[0][0]
    return float(min_value)


def get_avg(readings_df):
    avg_value = readings_df.agg({"value": "avg"}).collect()[0][0]
    return float(avg_value)


def print_traffic_light():
    for engine, color in traffic_light.items():
        border = "*" * (len(engine) + 4)

        # ANSI escape codes for changing text color
        color_code = ""
        if color == "green":
            color_code = "\033[92m"  # Green color
        elif color == "yellow":
            color_code = "\033[93m"  # Yellow color
        elif color == "red":
            color_code = "\033[91m"  # Red color

        # Print engine name with border and color
        print(border)
        print(f'* {color_code}{engine}\033[0m *')  # Reset color after engine name
        print(border)


if __name__ == "__main__":
    begin_of_analysis = "2024-02-09 08:00:00"
    for i in range(1, 11):
        engine_id = "engine_" + str(i)
        analyse_engine_data(engine_id, begin_of_analysis, False)
    print_traffic_light()


spark.stop()
