import pandas as pd
import numpy as np
from datetime import datetime

# Load Olist CSVs
df_products = pd.read_csv('data/olist_products_dataset.csv')
df_customers = pd.read_csv('data/olist_customers_dataset.csv')
df_orders = pd.read_csv('data/olist_orders_dataset.csv')
df_order_items = pd.read_csv('data/olist_order_items_dataset.csv')
df_reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
df_payments = pd.read_csv('data/olist_order_payments_dataset.csv')
df_sellers = pd.read_csv('data/olist_sellers_dataset.csv')
df_geo = pd.read_csv('data/olist_geolocation_dataset.csv')
df_cat_trans = pd.read_csv('data/product_category_name_translation.csv')

# --- PRODUCTS ---
# Merge product category translation for English names
products = df_products.merge(df_cat_trans, how='left', left_on='product_category_name', right_on='product_category_name')
products = products.rename(columns={
    'product_id': 'product_id',
    'product_category_name_english': 'category',
    'product_weight_g': 'weight',
})
products['weight'] = products['weight'].fillna(products['weight'].mean()) / 1000  # convert to kg
products['brand'] = 'Olist'  # Olist data has no brand info
products['is_prime'] = 0  # No prime info
products['stock_quantity'] = np.random.randint(10, 1000, size=len(products))
products['rating'] = np.nan  # Will fill from reviews later
products['name'] = products['category'].fillna('Unknown') + ' Item'
products['price'] = np.nan  # Will fill from order_items later

# --- CUSTOMERS ---
customers = df_customers.rename(columns={
    'customer_id': 'customer_id',
    'customer_unique_id': 'customer_unique_id',
    'customer_city': 'location',
    'customer_state': 'state',
})
customers['age_group'] = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], size=len(customers))
customers['income_level'] = np.random.choice(['Low', 'Medium', 'High'], size=len(customers), p=[0.3, 0.5, 0.2])
customers['tech_savvy'] = np.random.choice([0, 1], size=len(customers), p=[0.4, 0.6])
customers['prime_member'] = np.random.choice([0, 1], size=len(customers), p=[0.5, 0.5])
customers['avg_order_value'] = np.nan  # Will fill later
customers['loyalty_score'] = np.random.uniform(0, 100, size=len(customers))

# --- TRANSACTIONS ---
# Merge order_items with orders to get customer_id and order date
transactions = df_order_items.merge(df_orders[['order_id', 'customer_id', 'order_purchase_timestamp']], on='order_id', how='left')
# Merge with products for product details
transactions = transactions.merge(products[['product_id', 'category', 'weight']], on='product_id', how='left')
# Merge with customers for customer details
transactions = transactions.merge(customers[['customer_id', 'age_group', 'location', 'income_level', 'tech_savvy', 'prime_member', 'loyalty_score']], on='customer_id', how='left')
# Merge with reviews for satisfaction
transactions = transactions.merge(df_reviews[['order_id', 'review_score']], on='order_id', how='left')
# Merge with payments for payment value
transactions = transactions.merge(df_payments.groupby('order_id').agg({'payment_value': 'sum'}).reset_index(), on='order_id', how='left')

# Fill missing values and create required columns
transactions['quantity'] = transactions['order_item_id']
transactions['unit_price'] = transactions['price']
transactions['total_amount'] = transactions['price'] * transactions['quantity']
transactions['delivery_time'] = (pd.to_datetime(transactions['shipping_limit_date']) - pd.to_datetime(transactions['order_purchase_timestamp'])).dt.days
transactions['delivery_time'] = transactions['delivery_time'].fillna(transactions['delivery_time'].mean()).astype(int)
transactions['satisfaction_score'] = transactions['review_score'] * 20  # scale 1-5 to 0-100
transactions['satisfaction_category'] = pd.cut(transactions['satisfaction_score'], bins=[-1,59,79,100], labels=['Low','Medium','High'])
transactions['transaction_id'] = transactions['order_id'] + '_' + transactions['order_item_id'].astype(str)
transactions['transaction_date'] = pd.to_datetime(transactions['order_purchase_timestamp']).dt.strftime('%Y-%m-%d')
transactions['month'] = pd.to_datetime(transactions['order_purchase_timestamp']).dt.month
transactions['day_of_week'] = pd.to_datetime(transactions['order_purchase_timestamp']).dt.day_name()

# --- Fill in missing product prices and ratings ---
# Use average price and rating per product from transactions
grouped = transactions.groupby('product_id').agg({'unit_price': 'mean', 'review_score': 'mean'}).reset_index()
products = products.merge(grouped, on='product_id', how='left')
products['price'] = products['unit_price'].fillna(products['price'])
products['rating'] = products['review_score'].fillna(3.5)
products = products.drop(['unit_price', 'review_score'], axis=1)

# --- Fill in avg_order_value for customers ---
avg_order = transactions.groupby('customer_id').agg({'total_amount': 'mean'}).reset_index()
customers = customers.merge(avg_order, on='customer_id', how='left')
customers['avg_order_value'] = customers['total_amount'].fillna(0)
customers = customers.drop('total_amount', axis=1)

# --- Select and reorder columns to match your ML code ---
products_out = products[['product_id', 'name', 'category', 'brand', 'price', 'rating', 'weight', 'is_prime', 'stock_quantity']]
customers_out = customers[['customer_id', 'age_group', 'location', 'income_level', 'tech_savvy', 'prime_member', 'avg_order_value', 'loyalty_score']]
transactions_out = transactions[['transaction_id', 'customer_id', 'product_id', 'quantity', 'unit_price', 'total_amount', 'delivery_time', 'satisfaction_score', 'satisfaction_category', 'transaction_date', 'month', 'day_of_week']]

# --- Save to CSVs ---
products_out.to_csv('data/products.csv', index=False)
customers_out.to_csv('data/customers.csv', index=False)
transactions_out.to_csv('data/transactions.csv', index=False)

print('✅ Olist data prepared and saved as products.csv, customers.csv, transactions.csv!') 