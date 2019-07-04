from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import isnan, when, count, col

conf = SparkConf().setAppName("Read parquet")
ss = SparkSession.builder.config(conf=conf).getOrCreate()

# using SQLContext to read parquet file
sc = ss.sparkContext
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.parquet('/validation/validation_df.parquet')
df.show()
print("COunt ->>>>>>>>>>", df.count())
#print(df.filter((df["features"] == "") | df["features"].isNull()).count())
