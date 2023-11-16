from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import expr

# Create Spark session
spark = SparkSession.builder.appName('SentimentAnalysis').getOrCreate() 

# Load data from multiple JSON files as DataFrame
df = spark.read.json('gs://bigdata_yelp_review_test_train/smaller_json_file_*.json')

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

# Split the data into training, validation, and testing sets (60%, 20%, 20%)
train_data, rest_data = binary_features_df.randomSplit([0.6, 0.4], seed=42)
validation_data, test_data = rest_data.randomSplit([0.5, 0.5], seed=42)

# Train binary classification model on the training set
lsvc_binary = LinearSVC(maxIter=10, regParam=0.1)
binary_model = lsvc_binary.fit(train_data) 

# Make predictions on the validation set
validation_predictions = binary_model.transform(validation_data)

# Evaluate binary model on the validation set
evaluator = BinaryClassificationEvaluator()
validation_binary_accuracy = evaluator.evaluate(validation_predictions)
print("Binary Classification Area Under ROC on Validation Data:", validation_binary_accuracy)

# Make predictions on the testing set
test_predictions = binary_model.transform(test_data)

# Evaluate binary model on the testing set
test_binary_accuracy = evaluator.evaluate(test_predictions)
print("Binary Classification Area Under ROC on Test Data:", test_binary_accuracy)

# Save binary model to GCS
binary_model.save("gs://bigdata_yelp_review_test_train/sentiment_model_alternative")