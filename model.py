# Import Spark libraries
import pyspark.sql
import pyspark.ml
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import expr

# Create Spark session
spark = SparkSession.builder.appName('SentimentAnalysis').getOrCreate() 

# Load data as DataFrame
df = spark.read.json('gs://bigdata_yelp_review_test_train/smaller_json_file_1.json')

# Build text processing pipeline 
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="rawFeatures")
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

# Transform data
processed_df = pipeline.fit(df).transform(df)

# Convert multi-class to binary classification (positive or negative sentiment)
binary_df = processed_df.withColumn("label", expr("CASE WHEN stars >= 4 THEN 1 ELSE 0 END"))

# Extract features and binary label
binary_features_df = binary_df.select("features", "label")

# Train binary classification model
lsvc_binary = LinearSVC(maxIter=10, regParam=0.1)
binary_model = lsvc_binary.fit(binary_features_df) 

# Make predictions
predictions = binary_model.transform(binary_features_df)

# Evaluate binary model
evaluator = BinaryClassificationEvaluator()
binary_accuracy = evaluator.evaluate(predictions)
print("Binary Classification Area Under ROC:", binary_accuracy)

# Save binary model to GCS
#binary_model.write().overwrite().save("gs://bucket/sentiment_model_binary")
