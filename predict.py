import pandas as pd
import joblib

# Load the trained model
model = joblib.load('customer_purchase_predictor.joblib')

# Load necessary data
customers = pd.read_csv('input_data/customers.csv')
products = pd.read_csv('input_data/products.csv')
browsing_data = pd.read_csv('input_data/browsing_data.csv')

def predict_products_for_customer(customer_id, top_n=5):
    # Get customer data
    customer_data = customers[customers['customer_id'] == customer_id].iloc[0]
    
    # Prepare all products for prediction
    all_products = products.copy()
    all_products['customer_id'] = customer_id
    
    # Merge customer data
    all_products = pd.merge(all_products, customer_data.to_frame().T, on='customer_id')
    
    # Add browsing data
    browsing = browsing_data[browsing_data['customer_id'] == customer_id]
    browsing_agg = browsing.groupby('product_id').agg({
        'event_type': lambda x: (x == 'view').sum(),
        'time_spent': 'sum'
    }).reset_index()
    browsing_agg.columns = ['product_id', 'view_count', 'total_time_spent']
    all_products = pd.merge(all_products, browsing_agg, on='product_id', how='left')
    
    # Fill missing values
    all_products = all_products.fillna(0)
    all_products = all_products.infer_objects(copy=False)  # Address the FutureWarning
    
    # Add missing features
    all_products['days_since_last_purchase'] = customer_data.get('days_since_last_purchase', 0)
    all_products['total_spent'] = customer_data.get('total_spent', 0)
    all_products['avg_purchase_value'] = customer_data.get('avg_purchase_value', 0)
    all_products['purchase_count'] = customer_data.get('purchase_count', 0)
    
    # Prepare features for prediction
    features = ['age', 'customer_lifetime_value', 'price', 'view_count', 'total_time_spent',
                'gender', 'location', 'category', 'days_since_last_purchase', 'total_spent',
                'avg_purchase_value', 'purchase_count']
    X_pred = all_products[features]
    
    # Make predictions
    probabilities = model.predict_proba(X_pred)
    top_products_indices = probabilities.argsort()[0][-top_n:][::-1]
    
    # Select columns that are present in the DataFrame
    return_columns = ['product_id', 'name', 'category', 'price']
    return_columns = [col for col in return_columns if col in all_products.columns]
    
    return all_products.iloc[top_products_indices][return_columns]

# Example usage
customer_id = 1
print(f"\nTop 5 predicted products for customer {customer_id}:")
print(predict_products_for_customer(customer_id, top_n=5))
