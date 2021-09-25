from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName('Python apache_spark SQL basic example')\
        .config('spark.some.config.option', 'some-value')\
        .getOrCreate()

### Create json file using spark
sc = spark.sparkContext

# json 파일 읽어들이기
path = '/Users/~'
peopleDF = spark.read.json(path)

# printSchema() is able to see json schema structure
peopleDF.printSchema()
peopleDF.createOrReplaceTempView("people")


teenagerNamesDF = spark.sql("SELECT name FROM people WHERE age BETWEEN 13 AND 19")
teenagerNamesDF.show()