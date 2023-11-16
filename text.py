from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover, IDF
from pyspark.ml import Pipeline

# Assuming you have already created a Spark session
spark = SparkSession.builder.appName('SentimentAnalysisLoad').getOrCreate()

# Load the saved model
loaded_model = LinearSVCModel.load("gs://bigdata_yelp_review_test_train/sentiment_model_alternative")

# Check if the loaded model is an instance of LinearSVCModel
if isinstance(loaded_model, LinearSVCModel):
    print("Model loaded successfully!")
else:
    print("Failed to load the model. Check the model path and version compatibility.")

# Build text processing pipeline 
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="rawFeatures")
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")

# Create a pipeline with the tokenizer, stopwords remover, TF, IDF, and the loaded LinearSVC model
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, loaded_model])

# Sample text for prediction
sample_text = "The food here is bad, and the ambiance is worse"

# Create a DataFrame with the sample text
data = spark.createDataFrame([(sample_text,)], ["text"])

# Make predictions
predictions = pipeline.fit(data).transform(data)

# Display the result
predictions.select("text", "prediction").show()
