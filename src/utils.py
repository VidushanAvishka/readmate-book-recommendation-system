from pyspark.sql import SparkSession
import os

PYTHON_PATH = r"D:\GitHub\readmate-book-recommendation-system\venv\Scripts\python.exe"

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
# Ensure Hadoop native libraries are available to JVM on Windows
os.environ["PATH"] = os.path.join(os.environ["HADOOP_HOME"], "bin") + ";" + os.environ.get("PATH", "")


def create_spark_session(app_name="ReadMate Book Recommendation System"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.pyspark.python", PYTHON_PATH) \
        .config("spark.pyspark.driver.python", PYTHON_PATH) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark