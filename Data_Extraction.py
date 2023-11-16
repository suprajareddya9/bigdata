import functions_framework
import requests
import json
from google.cloud import pubsub_v1

def yelp_to_pubsub(request):
    # Define your Yelp Fusion API credentials
    client_id = '3C8py-b3W606H54OVHgH4g'
    api_key = 'ylZ9TmcoAy_kc7kWfSgKak07GzJf4ks0Dvys-hU4tT71N6-_leVAkkmPIXwJ8bon91dW00iCjwWhXpqdA7DyTRAGb3La2D46Vz3qph4yFmuilsfxGmMT8M1hpOU5ZXYx'

    # Define your search parameters
    search_params = {
        'term': 'Pizza',
        'location': 'New York, NY',
        'limit': 30  # You can adjust the number of results per request
    }

    # Define the base URL for Yelp Fusion API
    base_url = 'https://api.yelp.com/v3/businesses/search'

    # Set up the headers with your API Key
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    # Create a Pub/Sub client
    project_id = 'sapient-magnet-404702'
    topic_name = 'Yelp_Reviews_Raw'
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)

    yelp_reviews = []

    # Make the initial request to Yelp Fusion API
    response = requests.get(base_url, headers=headers, params=search_params)

    if response.status_code == 200:
        data = response.json()
        businesses = data.get('businesses', [])

        for business in businesses:
            business_name = business['name']
            business_id = business['id']

            # Make a request to get reviews for the business
            reviews_url = f'https://api.yelp.com/v3/businesses/{business_id}/reviews'
            reviews_response = requests.get(reviews_url, headers=headers)

            if reviews_response.status_code == 200:
                reviews_data = reviews_response.json()
                reviews = reviews_data.get('reviews', [])

                # Calculate the average rating for the business
                ratings = [review['rating'] for review in reviews]
                average_rating = sum(ratings) / len(ratings) if ratings else 0

                yelp_reviews.append({
                    'title': business_name,
                    'average_rating': average_rating,
                    'reviews': reviews
                })

    # Publish the collected data to Pub/Sub
    for review in yelp_reviews:
        message_data = json.dumps(review).encode('utf-8')
        publisher.publish(topic_path, message_data)

    return 'Data collection and publishing to Pub/Sub completed.'