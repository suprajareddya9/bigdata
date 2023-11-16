from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVCModel

# Assuming you have already created a Spark session
spark = SparkSession.builder.appName('SentimentAnalysisload').getOrCreate()

# Load the saved model
loaded_model = LinearSVCModel.load("gs://bigdata_yelp_review_test_train/sentiment_model_alternative")

# Check if the loaded model is an instance of LinearSVCModel
if isinstance(loaded_model, LinearSVCModel):
    print("Model loaded successfully!")
else:
    print("Failed to load the model. Check the model path and version compatibility.")

from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline

# Sample text for prediction
sample_text = "The food here is bad and ambience is worse"

# Assuming you have a tokenizer and a hashing term frequency (TF) feature in your original training pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")

# Create a pipeline with the tokenizer, hashing TF, and the loaded LinearSVC model
pipeline = Pipeline(stages=[tokenizer, hashing_tf, loaded_model])

# Create a DataFrame with the sample text
data = spark.createDataFrame([(sample_text,)], ["text"])

# Make predictions
predictions = pipeline.fit(data).transform(data)

# Display the result
predictions.select("text", "prediction").show()
