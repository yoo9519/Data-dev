from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName('Python Spark SQL basic example')\
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

# json 파일 읽어들이기
path = '/Users/younghun/Desktop/gitrepo/TIL/pyspark/people.json'
df = spark.read.json(path)

# Global Temporary View 생성
df.createOrReplaceGlobalTempView('people')

# 'global_temp' 라는 키워드 꼭 붙여주자!
sqlDF = spark.sql('SELECT * FROM global_temp.people')
sqlDF.show()