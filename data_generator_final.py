from flask import Flask
import uuid
import random
import pandas as pd
from tqdm import tqdm

app = Flask(__name__)

total_subscribers = 10000
active_subscribers = int(total_subscribers * 0.8)
inactive_subscribers = total_subscribers - active_subscribers

subscriber_data_list = []
test_data_list = []

# Generate active subscribers with tightly clustered characteristics
for _ in tqdm(range(active_subscribers)):
    subscriber_id = str(uuid.uuid4())
    engagement_frequency = random.choice([1, 2])  # Numeric values for engagement frequency
    interaction_recency = random.randint(1, 5)
    subscription_data = {
        "subscription_status": 1,  # Numeric value for active subscription status
        "cancellationRequested": False,
        "pauseRequested": False,
        "onHold": False,
        "expired": False,
    }
    subscriber_data = {
        "subscriber_id": subscriber_id,
        "engagement_frequency": engagement_frequency,
        "subscription_age": random.randint(30, 180),
        "billing_history": 1,  # Numeric value for good billing history
        "content_preferences": random.choice([1, 2]),  # Numeric values for content preferences
        "interaction_recency": interaction_recency,
        "usage_patterns": 1,  # Numeric value for high usage patterns
        "feedback_ratings": 1,  # Numeric value for positive feedback ratings
        "device_preference": 1,  # Numeric value for mobile device preference
        "social_interactions": 1,  # Numeric value for high social interactions
        "subscription_plan": 2,  # Numeric value for premium subscription plan
        "promotions_offers": 1,  # Numeric value for promotions and offers
        "customer_service_interactions": 2,  # Numeric value for low customer service interactions
        "competing_services_usage": 2,  # Numeric value for low competing services usage
        "geographic_location": 1,  # Numeric value for urban geographic location
        "email_engagement": 1,  # Numeric value for high email engagement
        **subscription_data,
    }
    subscriber_data_list.append(subscriber_data)

# Generate non-active subscribers with tightly clustered characteristics for churn prediction
for _ in tqdm(range(inactive_subscribers)):
    subscriber_id = str(uuid.uuid4())
    engagement_frequency = 3  # Numeric value for low engagement frequency
    interaction_recency = random.randint(5, 10)
    subscription_data = {
        "subscription_status": random.choice([3, 4]),  # Numeric values for churned or paused subscription status
        "cancellationRequested": True,
        "pauseRequested": True,
        "onHold": True,
        "expired": False,
    }
    subscriber_data = {
        "subscriber_id": subscriber_id,
        "engagement_frequency": engagement_frequency,
        "subscription_age": random.randint(150, 365),
        "billing_history": 2,  # Numeric value for poor billing history
        "content_preferences": 3,  # Numeric value for sports content preferences
        "interaction_recency": interaction_recency,
        "usage_patterns": 2,  # Numeric value for low usage patterns
        "feedback_ratings": 3,  # Numeric value for negative feedback ratings
        "device_preference": 3,  # Numeric value for tablet device preference
        "social_interactions": 2,  # Numeric value for low social interactions
        "subscription_plan": 1,  # Numeric value for basic subscription plan
        "promotions_offers": 1,  # Numeric value for promotions and offers
        "customer_service_interactions": 1,  # Numeric value for medium customer service interactions
        "competing_services_usage": 1,  # Numeric value for medium competing services usage
        "geographic_location": 3,  # Numeric value for rural geographic location
        "email_engagement": 2,  # Numeric value for low email engagement
        **subscription_data,
    }
    subscriber_data_list.append(subscriber_data)

# Create DataFrames from the list of subscriber data
df = pd.DataFrame(subscriber_data_list)

# Filter subscribers who have not requested cancellation for test data
test_data_list = [subscriber for subscriber in subscriber_data_list if not subscriber["cancellationRequested"]]

# Create a DataFrame from the filtered test data
test_df = pd.DataFrame(test_data_list)

# Write the DataFrames to CSV files
csv_filename = "subscriber_data.csv"
test_csv_filename = "test_data.csv"
df.to_csv(csv_filename, index=False)
test_df.to_csv(test_csv_filename, index=False)

if __name__ == "__main__":
    app.run()
