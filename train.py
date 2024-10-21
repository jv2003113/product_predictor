import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the data
customers = pd.read_csv('input_data/customers.csv')
products = pd.read_csv('input_data/products.csv')
purchases = pd.read_csv('input_data/purchases.csv')
browsing_data = pd.read_csv('input_data/browsing_data.csv')

# Merge data
df = purchases.merge(customers, on='customer_id')
df = df.merge(products, on='product_id')
df = df.merge(browsing_data, on=['customer_id', 'product_id'], how='left')

# Feature engineering
df['purchase_count'] = df.groupby('customer_id')['purchase_id'].transform('count')
df['total_spent'] = df.groupby('customer_id')['price'].transform('sum')
df['avg_purchase_value'] = df['total_spent'] / df['purchase_count']
df['days_since_last_purchase'] = (pd.to_datetime(df['purchase_date'].max()) - pd.to_datetime(df['purchase_date'])).dt.days

# Aggregate browsing data
browsing_agg = browsing_data.groupby(['customer_id', 'product_id']).agg({
    'event_type': lambda x: (x == 'view').sum(),
    'time_spent': 'sum'
}).reset_index()
browsing_agg.columns = ['customer_id', 'product_id', 'view_count', 'total_time_spent']

df = df.merge(browsing_agg, on=['customer_id', 'product_id'], how='left')

# Prepare features and target
features = ['age', 'customer_lifetime_value', 'price', 'purchase_count', 'total_spent', 
            'avg_purchase_value', 'days_since_last_purchase', 'view_count', 'total_time_spent']
categorical_features = ['gender', 'location', 'category']
target = 'product_id'

X = df[features + categorical_features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing steps
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(clf, 'customer_purchase_predictor.joblib')

print("Model training complete. Model saved as 'customer_purchase_predictor.joblib'")

# Function to predict products for a given customer
def predict_products(customer_id, top_n=5):
    # Get customer data
    customer_data = df[df['customer_id'] == customer_id].iloc[0]
    
    # Get all products
    all_products = products.copy()
    all_products['customer_id'] = customer_id
    
    # Merge with customer data
    all_products = all_products.merge(customers[customers['customer_id'] == customer_id], on='customer_id')
    
    # Add missing features
    all_products['purchase_count'] = customer_data['purchase_count']
    all_products['total_spent'] = customer_data['total_spent']
    all_products['avg_purchase_value'] = customer_data['avg_purchase_value']
    all_products['days_since_last_purchase'] = customer_data['days_since_last_purchase']
    
    # Add browsing data (if available)
    browsing = browsing_data[(browsing_data['customer_id'] == customer_id)]
    if not browsing.empty:
        browsing_agg = browsing.groupby('product_id').agg({
            'event_type': lambda x: (x == 'view').sum(),
            'time_spent': 'sum'
        }).reset_index()
        browsing_agg.columns = ['product_id', 'view_count', 'total_time_spent']
        all_products = all_products.merge(browsing_agg, on='product_id', how='left')
    else:
        all_products['view_count'] = 0
        all_products['total_time_spent'] = 0
    
    # Fill missing values
    all_products = all_products.fillna(0)
    
    # Prepare features for prediction
    X_pred = all_products[features + categorical_features]
    
    # Make predictions
    probabilities = clf.predict_proba(X_pred)
    top_products_indices = np.argsort(probabilities, axis=1)[0][-top_n:][::-1]
    
    return products.iloc[top_products_indices]

# Example usage
print("\nTop 5 predicted products for customer 1:")
print(predict_products(1, top_n=5))
