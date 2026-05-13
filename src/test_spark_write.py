from utils import create_spark_session

spark = create_spark_session("Spark Write Test")

df = spark.createDataFrame(
    [(1, "hello"), (2, "spark")],
    ["id", "text"]
)

df.write.mode("overwrite").csv("outputs/test_write")

print("SUCCESS")

spark.stop()