import os

os.environ["PYSPARK_PYTHON"] = r"D:\GitHub\readmate-book-recommendation-system\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"D:\GitHub\readmate-book-recommendation-system\venv\Scripts\python.exe"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("HelloSpark") \
    .master("local[*]") \
    .getOrCreate()

data = [("Avishka", 1), ("Book", 2)]
columns = ["name", "id"]

df = spark.createDataFrame(data, columns)
df.show()

spark.stop()