import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate customer data
def generate_customers(num_customers):
    return pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'age': np.random.randint(18, 80, num_customers),
        'gender': np.random.choice(['M', 'F'], num_customers),
        'location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], num_customers),
        'account_creation_date': pd.date_range(end=datetime.now(), periods=num_customers).tolist(),
        'customer_lifetime_value': np.random.uniform(100, 10000, num_customers).round(2)
    })

# Generate product data
def generate_products(num_products):
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Sports']
    return pd.DataFrame({
        'product_id': range(1, num_products + 1),
        'category': np.random.choice(categories, num_products),
        'price': np.random.uniform(10, 1000, num_products).round(2),
        'description': [f'Product {i} description' for i in range(1, num_products + 1)]
    })

# Generate purchase data
def generate_purchases(customers, products, num_purchases):
    return pd.DataFrame({
        'purchase_id': range(1, num_purchases + 1),
        'customer_id': np.random.choice(customers['customer_id'], num_purchases),
        'product_id': np.random.choice(products['product_id'], num_purchases),
        'purchase_date': pd.date_range(end=datetime.now(), periods=num_purchases).tolist(),
        'quantity': np.random.randint(1, 5, num_purchases)
    })

# Generate browsing data
def generate_browsing_data(customers, products, num_events):
    return pd.DataFrame({
        'event_id': range(1, num_events + 1),
        'customer_id': np.random.choice(customers['customer_id'], num_events),
        'product_id': np.random.choice(products['product_id'], num_events),
        'event_type': np.random.choice(['view', 'add_to_cart', 'remove_from_cart'], num_events),
        'event_date': pd.date_range(end=datetime.now(), periods=num_events).tolist(),
        'time_spent': np.random.uniform(5, 300, num_events).round(2)  # time spent in seconds
    })

# Main function to generate all data
def generate_all_data(num_customers=1000, num_products=100, num_purchases=5000, num_browsing_events=10000):
    customers = generate_customers(num_customers)
    products = generate_products(num_products)
    purchases = generate_purchases(customers, products, num_purchases)
    browsing_data = generate_browsing_data(customers, products, num_browsing_events)
    
    return customers, products, purchases, browsing_data

# Generate the data
customers, products, purchases, browsing_data = generate_all_data()

# Save to CSV files
customers.to_csv('customers.csv', index=False)
products.to_csv('products.csv', index=False)
purchases.to_csv('purchases.csv', index=False)
browsing_data.to_csv('browsing_data.csv', index=False)

print("Data generation complete. Files saved: customers.csv, products.csv, purchases.csv, browsing_data.csv")
