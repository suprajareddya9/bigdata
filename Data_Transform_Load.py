#!/usr/bin/env python

"""BigQuery I/O PySpark example."""

import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
nltk.download('punkt')
nltk.download('stopwords')
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode
#import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, DoubleType
from google.cloud import bigquery
# Import Spark libraries
import pyspark.sql
import pyspark.ml
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF , CountVectorizer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import expr
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.ml import PipelineModel

spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('spark-bigquery-demo') \
  .getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector.
bucket = "testing1239"
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from BigQuery.
reviews = spark.read.format('bigquery') \
  .option('table', 'sapient-magnet-404702.yelp_reviews.reviews1') \
  .load()

reviews.createOrReplaceTempView('reviews')

df = spark.sql(
    'SELECT data FROM reviews')
df.show()

# Select the column you want to convert to a list of strings
column_to_convert = df.select("data")

# Convert the column to a list of strings
list_of_strings = [row["data"] for row in column_to_convert.collect()]

processed_data = []

# Define function for text preprocessing
def preprocess_text(text):
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Elimination of stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Process each JSON string in the list
for json_string in list_of_strings:
    # Parse JSON string
    data = json.loads(json_string)

    title = data['title']
    average_rating = data['average_rating']

    # Process each review
    for review in data['reviews']:
        processed_text = preprocess_text(review['text'])
        processed_data.append({
            'title': title,
            'average_rating': average_rating,
            'User_ID': review['id'],
            'name': review['user']['name'],
            'text': processed_text,
            'rating': review['rating']
        })

# Create a DataFrame
df_bq = pd.DataFrame(processed_data)

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df_bq)
spark_df.show()

'''
# Load data from multiple JSON files as DataFrame
df_ml = spark.read.json('gs://bigdata_yelp_review_test_train/smaller_json_file_4.json')

# Create a new column 'label' based on the 'stars' column
df_new = df_ml.withColumn('label', expr("CASE WHEN stars > 3 THEN 1.0 ELSE 0.0 END"))

# Select only the relevant columns
df_new = df_new.select('text', 'label')

# Tokenize, remove stop words, and create a feature vector
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")

# Create a linear SVM model
svm = LinearSVC(featuresCol="features", labelCol="label", predictionCol="prediction")

# Create a pipeline
pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, svm])

# Fit the model
model = pipeline.fit(df_new)
'''

loaded_model = PipelineModel.load("gs://bigdata_yelp_review_test_train/svc")

predictions = loaded_model.transform(spark_df)
predictions.show()

columns = ['title', 'User_ID', 'name', 'text', 'rating', 'prediction']
final = predictions.select(columns)

final = final.withColumn(
    "prediction",
    F.when(final["prediction"] == 0, "Negative").when(final["prediction"] == 1, "Positive").otherwise("Unknown")
)

final.show()

final.write.format('bigquery').option('table', 'sapient-magnet-404702.yelp_reviews.yelp_analysis').mode('append').save()

#model.save("gs://bigdata_yelp_review_test_train/svc")

client = bigquery.Client()
table_id = 'sapient-magnet-404702.yelp_reviews.reviews1'

# Construct the DELETE statement
sql_query = f"DELETE FROM `{table_id}` WHERE TRUE"

query_job = client.query(sql_query)
query_job.result()