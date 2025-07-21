#!/usr/bin/env python3
"""
🛒 SMARTMARKET CLI - Machine Learning Powered Marketplace
A complete e-commerce CLI application demonstrating all 4 ML concepts:
1. Regression - Price Prediction & Sales Forecasting
2. Classification - Customer Satisfaction Prediction
3. Clustering - Customer Segmentation
4. Association Rules - Product Recommendations

Author: ML Learning Project
Version: 1.0
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
from itertools import combinations

from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

class DataGenerator:
    """Generate realistic marketplace datasets"""
    
    def __init__(self, random_state=42):
        np.random.seed(random_state)
        self.random_state = random_state
    
    def generate_products(self, n_products=500):
        """Generate product catalog"""
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Health', 'Toys']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'Generic']
        
        products = []
        for i in range(n_products):
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            
            # Category-specific price ranges
            if category == 'Electronics':
                base_price = np.random.uniform(50, 1000)
            elif category == 'Clothing':
                base_price = np.random.uniform(15, 200)
            elif category == 'Home & Garden':
                base_price = np.random.uniform(20, 500)
            elif category == 'Sports':
                base_price = np.random.uniform(25, 300)
            elif category == 'Books':
                base_price = np.random.uniform(10, 50)
            elif category == 'Health':
                base_price = np.random.uniform(15, 150)
            else:  # Toys
                base_price = np.random.uniform(10, 100)
            
            # Brand premium
            brand_multiplier = 1.0
            if brand in ['BrandA', 'BrandB']:
                brand_multiplier = 1.3
            elif brand in ['BrandC', 'BrandD']:
                brand_multiplier = 1.1
            
            price = base_price * brand_multiplier
            
            # Product features
            rating = np.random.uniform(2.5, 5.0)
            weight = np.random.uniform(0.1, 10.0)  # kg
            is_prime = np.random.choice([0, 1], p=[0.6, 0.4])
            stock_quantity = np.random.randint(0, 1000)
            
            products.append({
                'product_id': f'P{i+1:04d}',
                'name': f'{brand} {category} Item {i+1}',
                'category': category,
                'brand': brand,
                'price': round(price, 2),
                'rating': round(rating, 1),
                'weight': round(weight, 2),
                'is_prime': is_prime,
                'stock_quantity': stock_quantity
            })
        
        return pd.DataFrame(products)
    
    def generate_customers(self, n_customers=1000):
        """Generate customer data"""
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        locations = ['Urban', 'Suburban', 'Rural']
        income_levels = ['Low', 'Medium', 'High']
        
        customers = []
        for i in range(n_customers):
            age_group = np.random.choice(age_groups, p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05])
            location = np.random.choice(locations, p=[0.4, 0.45, 0.15])
            income = np.random.choice(income_levels, p=[0.3, 0.5, 0.2])
            
            # Customer behavior patterns
            if age_group in ['18-25', '26-35']:
                tech_savvy = np.random.choice([0, 1], p=[0.2, 0.8])
                prime_member = np.random.choice([0, 1], p=[0.3, 0.7])
            else:
                tech_savvy = np.random.choice([0, 1], p=[0.6, 0.4])
                prime_member = np.random.choice([0, 1], p=[0.5, 0.5])
            
            # Spending patterns
            if income == 'High':
                avg_order_value = np.random.normal(150, 50)
            elif income == 'Medium':
                avg_order_value = np.random.normal(75, 25)
            else:
                avg_order_value = np.random.normal(35, 15)
            
            avg_order_value = max(10, avg_order_value)  # Minimum order value
            
            customers.append({
                'customer_id': f'C{i+1:04d}',
                'age_group': age_group,
                'location': location,
                'income_level': income,
                'tech_savvy': tech_savvy,
                'prime_member': prime_member,
                'avg_order_value': round(avg_order_value, 2),
                'loyalty_score': np.random.uniform(0, 100)
            })
        
        return pd.DataFrame(customers)
    
    def generate_transactions(self, products_df, customers_df, n_transactions=5000):
        """Generate transaction data"""
        transactions = []
        
        for i in range(n_transactions):
            customer = customers_df.sample(1).iloc[0]
            
            # Number of items in transaction
            if customer['prime_member']:
                n_items = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.07, 0.03])
            else:
                n_items = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            
            # Select products
            selected_products = products_df.sample(n_items)
            
            for _, product in selected_products.iterrows():
                # Quantity
                quantity = np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
                
                # Price adjustments (discounts, etc.)
                price_multiplier = np.random.uniform(0.8, 1.0)  # Some discounts
                final_price = product['price'] * price_multiplier
                
                # Delivery time (influenced by location, prime membership, weight)
                base_delivery_time = 5  # days
                if customer['prime_member']:
                    base_delivery_time = 2
                if customer['location'] == 'Rural':
                    base_delivery_time += 2
                if product['weight'] > 5:
                    base_delivery_time += 1
                
                delivery_time = base_delivery_time + np.random.randint(-1, 3)
                delivery_time = max(1, delivery_time)
                
                # Customer satisfaction (influenced by multiple factors)
                satisfaction_score = 70  # Base satisfaction
                
                if delivery_time <= 3:
                    satisfaction_score += 15
                elif delivery_time > 7:
                    satisfaction_score -= 20
                
                if product['rating'] >= 4.5:
                    satisfaction_score += 10
                elif product['rating'] < 3:
                    satisfaction_score -= 15
                
                if final_price < product['price'] * 0.9:  # Good discount
                    satisfaction_score += 10
                
                satisfaction_score += np.random.normal(0, 10)
                satisfaction_score = np.clip(satisfaction_score, 0, 100)
                
                # Satisfaction category
                if satisfaction_score >= 80:
                    satisfaction_category = 'High'
                elif satisfaction_score >= 60:
                    satisfaction_category = 'Medium'
                else:
                    satisfaction_category = 'Low'
                
                transaction_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
                
                transactions.append({
                    'transaction_id': f'T{len(transactions)+1:06d}',
                    'customer_id': customer['customer_id'],
                    'product_id': product['product_id'],
                    'quantity': quantity,
                    'unit_price': round(final_price, 2),
                    'total_amount': round(final_price * quantity, 2),
                    'delivery_time': delivery_time,
                    'satisfaction_score': round(satisfaction_score, 1),
                    'satisfaction_category': satisfaction_category,
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'month': transaction_date.month,
                    'day_of_week': transaction_date.strftime('%A')
                })
        
        return pd.DataFrame(transactions)

class MarketplaceML:
    """Main ML analysis class"""
    
    def __init__(self):
        self.products_df = None
        self.customers_df = None
        self.transactions_df = None
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        """Load or generate marketplace data"""
        print("🔄 Loading marketplace data...")
        
        # Check if data files exist
        if (os.path.exists('data/products.csv') and 
            os.path.exists('data/customers.csv') and 
            os.path.exists('data/transactions.csv')):
            
            print("📁 Loading existing data files...")
            self.products_df = pd.read_csv('data/products.csv')
            self.customers_df = pd.read_csv('data/customers.csv')
            self.transactions_df = pd.read_csv('data/transactions.csv')
        else:
            print("🏗️ Generating new marketplace data...")
            generator = DataGenerator()
            
            self.products_df = generator.generate_products(500)
            self.customers_df = generator.generate_customers(1000)
            self.transactions_df = generator.generate_transactions(
                self.products_df, self.customers_df, 5000
            )
            
            # Save data
            self.products_df.to_csv('products.csv', index=False)
            self.customers_df.to_csv('customers.csv', index=False)
            self.transactions_df.to_csv('transactions.csv', index=False)
            
            print("💾 Data saved to CSV files!")
        
        print(f"✅ Data loaded successfully!")
        print(f"   📦 Products: {len(self.products_df)}")
        print(f"   👥 Customers: {len(self.customers_df)}")
        print(f"   🛒 Transactions: {len(self.transactions_df)}")
    
    def regression_analysis(self):
        """Regression: Predict delivery time and sales volume"""
        print("\n📈 REGRESSION ANALYSIS")
        print("=" * 50)
        
        # Merge data for analysis
        merged_df = self.transactions_df.merge(self.products_df, on='product_id')
        merged_df = merged_df.merge(self.customers_df, on='customer_id')
        
        print("🎯 Task 1: Predict Delivery Time")
        print("-" * 30)
        
        # Prepare features for delivery time prediction
        feature_columns = ['weight', 'prime_member', 'total_amount']
        
        # Create binary features
        merged_df['location_rural'] = (merged_df['location'] == 'Rural').astype(int)
        merged_df['location_urban'] = (merged_df['location'] == 'Urban').astype(int)
        
        feature_columns.extend(['location_rural', 'location_urban'])
        
        X = merged_df[feature_columns]
        y = merged_df['delivery_time']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"   RMSE: {rmse:.2f} days")
        print(f"   R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Most Important Factors for Delivery Time:")
        for _, row in feature_importance.head(3).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save model
        self.models['delivery_time'] = model
        self.scalers['delivery_time'] = scaler
        
        # Predict delivery time for new order
        print(f"\n🔮 Delivery Time Prediction Example:")
        sample_order = {
            'weight': 2.5,
            'prime_member': 1,
            'total_amount': 85.50,
            'location_rural': 0,
            'location_urban': 1
        }
        
        sample_features = np.array([[sample_order[col] for col in feature_columns]])
        sample_scaled = scaler.transform(sample_features)
        predicted_time = model.predict(sample_scaled)[0]
        
        print(f"   Order details: Prime member, Urban area, 2.5kg, $85.50")
        print(f"   Predicted delivery time: {predicted_time:.1f} days")
        
        print("\n🎯 Task 2: Predict Monthly Sales Volume")
        print("-" * 30)
        
        # Aggregate sales by month and product category
        monthly_sales = merged_df.groupby(['month', 'category']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        # Simple time series features
        monthly_sales['month_sin'] = np.sin(2 * np.pi * monthly_sales['month'] / 12)
        monthly_sales['month_cos'] = np.cos(2 * np.pi * monthly_sales['month'] / 12)
        
        # Category encoding
        category_encoder = LabelEncoder()
        monthly_sales['category_encoded'] = category_encoder.fit_transform(monthly_sales['category'])
        
        # Predict quantity
        X_sales = monthly_sales[['month_sin', 'month_cos', 'category_encoded']]
        y_sales = monthly_sales['quantity']
        
        sales_model = LinearRegression()
        sales_model.fit(X_sales, y_sales)
        
        # Predict next month's sales
        next_month = (datetime.now().month % 12) + 1
        predictions = []
        
        for category in monthly_sales['category'].unique():
            cat_encoded = category_encoder.transform([category])[0]
            features = np.array([[
                np.sin(2 * np.pi * next_month / 12),
                np.cos(2 * np.pi * next_month / 12),
                cat_encoded
            ]])
            pred_quantity = sales_model.predict(features)[0]
            predictions.append((category, max(0, pred_quantity)))
        
        print(f"📅 Predicted sales for next month:")
        for category, quantity in sorted(predictions, key=lambda x: x[1], reverse=True)[:3]:
            print(f"   {category}: {quantity:.0f} units")
        
        self.models['sales_model'] = sales_model
        self.scalers['category_encoder'] = category_encoder
    
    def classification_analysis(self):
        """Classification: Predict customer satisfaction"""
        print("\n🏷️ CLASSIFICATION ANALYSIS")
        print("=" * 50)
        
        # Merge data
        merged_df = self.transactions_df.merge(self.products_df, on='product_id')
        merged_df = merged_df.merge(self.customers_df, on='customer_id')
        
        print("🎯 Task: Predict Customer Satisfaction Level")
        print("-" * 45)
        
        # Prepare features
        feature_columns = [
            'delivery_time', 'unit_price', 'rating', 'prime_member', 
            'tech_savvy', 'loyalty_score'
        ]
        
        # Add categorical features
        merged_df['category_electronics'] = (merged_df['category'] == 'Electronics').astype(int)
        merged_df['category_clothing'] = (merged_df['category'] == 'Clothing').astype(int)
        merged_df['income_high'] = (merged_df['income_level'] == 'High').astype(int)
        
        feature_columns.extend(['category_electronics', 'category_clothing', 'income_high'])
        
        X = merged_df[feature_columns].fillna(0)  # Patch: fill NaNs with 0
        y = merged_df['satisfaction_category']
        
        # Remove rows where y is NaN
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Class distribution
        print(f"\n📋 Satisfaction Distribution:")
        for category in ['High', 'Medium', 'Low']:
            count = sum(y_test == category)
            percentage = count / len(y_test) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Most Important Factors for Satisfaction:")
        for _, row in feature_importance.head(4).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save model
        self.models['satisfaction'] = model
        self.scalers['satisfaction'] = scaler
        
        # Predict satisfaction for new customer
        print(f"\n🔮 Satisfaction Prediction Example:")
        sample_customer = {
            'delivery_time': 3,
            'unit_price': 45.99,
            'rating': 4.2,
            'prime_member': 1,
            'tech_savvy': 1,
            'loyalty_score': 75.5,
            'category_electronics': 1,
            'category_clothing': 0,
            'income_high': 0
        }
        
        sample_features = pd.DataFrame([sample_customer])[feature_columns]
        sample_scaled = scaler.transform(sample_features)
        pred_satisfaction = model.predict(sample_scaled)[0]
        pred_proba = model.predict_proba(sample_scaled)[0]
        
        print(f"   Customer: Prime member, tech-savvy, electronics purchase")
        print(f"   Predicted satisfaction: {pred_satisfaction}")
        
        # Show probabilities
        classes = model.classes_
        print(f"   Confidence scores:")
        for i, cls in enumerate(classes):
            print(f"     {cls}: {pred_proba[i]:.2f}")
    
    def clustering_analysis(self):
        """Clustering: Customer segmentation"""
        print("\n👥 CLUSTERING ANALYSIS") 
        print("=" * 50)
        
        print("🎯 Task: Customer Segmentation")
        print("-" * 30)
        
        # Aggregate customer behavior
        customer_behavior = self.transactions_df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'satisfaction_score': 'mean',
            'delivery_time': 'mean'
        }).reset_index()
        
        # Flatten column names
        customer_behavior.columns = [
            'customer_id', 'total_spent', 'avg_order_value', 'order_frequency',
            'avg_satisfaction', 'avg_delivery_time'
        ]
        
        # Merge with customer data
        customer_features = customer_behavior.merge(self.customers_df, on='customer_id')
        
        # Patch: Ensure 'avg_order_value' exists
        if 'avg_order_value' not in customer_features.columns:
            customer_features['avg_order_value'] = customer_features['total_spent'] / customer_features['order_frequency']
        
        # Prepare features for clustering
        clustering_features = [
            'total_spent', 'avg_order_value', 'order_frequency', 
            'avg_satisfaction', 'loyalty_score'
        ]
        
        X_cluster = customer_features[clustering_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Determine optimal number of clusters
        inertias = []
        silhouette_scores = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        # Choose best k (highest silhouette score)
        best_k = K_range[np.argmax(silhouette_scores)]
        
        print(f"🔍 Optimal number of customer segments: {best_k}")
        print(f"   Silhouette score: {max(silhouette_scores):.3f}")
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        customer_features['cluster'] = cluster_labels
        
        # Analyze segments
        print(f"\n📊 Customer Segments Analysis:")
        for cluster in range(best_k):
            cluster_customers = customer_features[customer_features['cluster'] == cluster]
            size = len(cluster_customers)
            
            print(f"\n🔸 Segment {cluster + 1} ({size} customers):")
            print(f"   Average total spent: ${cluster_customers['total_spent'].mean():.2f}")
            print(f"   Average order value: ${cluster_customers['avg_order_value'].mean():.2f}")
            print(f"   Average order frequency: {cluster_customers['order_frequency'].mean():.1f}")
            print(f"   Average satisfaction: {cluster_customers['avg_satisfaction'].mean():.1f}")
            print(f"   Average loyalty score: {cluster_customers['loyalty_score'].mean():.1f}")
            
            # Dominant characteristics
            prime_pct = cluster_customers['prime_member'].mean() * 100
            high_income_pct = (cluster_customers['income_level'] == 'High').mean() * 100
            tech_savvy_pct = cluster_customers['tech_savvy'].mean() * 100
            
            print(f"   Prime members: {prime_pct:.1f}%")
            print(f"   High income: {high_income_pct:.1f}%")
            print(f"   Tech savvy: {tech_savvy_pct:.1f}%")
            
            # Segment characterization
            if cluster_customers['total_spent'].mean() > customer_features['total_spent'].mean() * 1.5:
                if cluster_customers['order_frequency'].mean() > customer_features['order_frequency'].mean():
                    segment_type = "💎 VIP Customers (High value, frequent)"
                else:
                    segment_type = "🛍️ Big Spenders (High value, occasional)"
            elif cluster_customers['order_frequency'].mean() > customer_features['order_frequency'].mean():
                segment_type = "🔄 Regular Customers (Frequent, moderate value)"
            else:
                segment_type = "🌱 Casual Customers (Low frequency, low value)"
            
            print(f"   Profile: {segment_type}")
        
        # Save model
        self.models['clustering'] = kmeans
        self.scalers['clustering'] = scaler
        
        # Predict segment for new customer
        print(f"\n🔮 Customer Segmentation Example:")
        sample_customer = {
            'total_spent': 450.00,
            'avg_order_value': 75.00,
            'order_frequency': 6,
            'avg_satisfaction': 82.5,
            'loyalty_score': 68.5
        }
        
        sample_features = np.array([[sample_customer[col] for col in clustering_features]])
        sample_scaled = scaler.transform(sample_features)
        predicted_segment = kmeans.predict(sample_scaled)[0]
        
        print(f"   Customer profile: 6 orders, $75 avg order, $450 total spent")
        print(f"   Predicted segment: {predicted_segment + 1}")
    
    def association_rules_analysis(self):
        """Association Rules: Product recommendations"""
        print("\n🛒 ASSOCIATION RULES ANALYSIS")
        print("=" * 50)
        
        print("🎯 Task: Find Products Bought Together")
        print("-" * 38)
        
        # Create market basket data
        # Group transactions by customer and date to find baskets
        basket_transactions = self.transactions_df.groupby([
            'customer_id', 'transaction_date'
        ])['product_id'].apply(list).reset_index()
        
        basket_transactions['basket_size'] = basket_transactions['product_id'].apply(len)
        
        # Filter for baskets with multiple items
        multi_item_baskets = basket_transactions[
            basket_transactions['basket_size'] >= 2
        ]['product_id'].tolist()
        
        print(f"📊 Market Basket Statistics:")
        print(f"   Total shopping baskets: {len(basket_transactions)}")
        print(f"   Multi-item baskets: {len(multi_item_baskets)}")
        print(f"   Average basket size: {basket_transactions['basket_size'].mean():.1f}")
        
        # Get product names for better readability
        product_names = dict(zip(self.products_df['product_id'], self.products_df['name']))
        
        # Calculate item frequencies
        all_items = [item for basket in multi_item_baskets for item in basket]
        item_counts = pd.Series(all_items).value_counts()
        
        print(f"\n📈 Most Popular Products:")
        for i, (product_id, count) in enumerate(item_counts.head(5).items()):
            product_name = product_names.get(product_id, product_id)[:30]
            frequency = count / len(multi_item_baskets)
            print(f"   {i+1}. {product_name}: {count} times ({frequency:.1%})")
        
        # Simple association rule mining
        def calculate_support(itemset, baskets):
            count = sum(1 for basket in baskets if all(item in basket for item in itemset))
            return count / len(baskets)
        
        def calculate_confidence(antecedent, consequent, baskets):
            antecedent_support = calculate_support(antecedent, baskets)
            if antecedent_support == 0:
                return 0
            rule_support = calculate_support(antecedent + consequent, baskets)
            return rule_support / antecedent_support
        
        def calculate_lift(antecedent, consequent, baskets):
            confidence = calculate_confidence(antecedent, consequent, baskets)
            consequent_support = calculate_support(consequent, baskets)
            if consequent_support == 0:
                return 0
            return confidence / consequent_support
        
        # Find association rules
        min_support = 0.05  # 5%
        min_confidence = 0.3  # 30%
        
        rules = []
        frequent_items = item_counts[item_counts >= len(multi_item_baskets) * min_support].index.tolist()
        
        print(f"\n🔍 Finding Association Rules...")
        print(f"   Minimum support: {min_support:.1%}")
        print(f"   Minimum confidence: {min_confidence:.1%}")
        
        # Check pairs of frequent items
        for item1 in frequent_items[:20]:  # Limit for performance
            for item2 in frequent_items[:20]:
                if item1 != item2:
                    support = calculate_support([item1, item2], multi_item_baskets)
                    if support >= min_support:
                        confidence = calculate_confidence([item1], [item2], multi_item_baskets)
                        if confidence >= min_confidence:
                            lift = calculate_lift([item1], [item2], multi_item_baskets)
                            rules.append({
                                'antecedent': item1,
                                'consequent': item2,
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
        
        # Sort rules by lift, ensure type safety
        rules.sort(key=lambda x: float(x['lift']), reverse=True)
        
        print(f"\n📋 Top Association Rules Found:")
        if rules:
            for i, rule in enumerate(rules[:5]):
                ant_name = product_names.get(rule['antecedent'], rule['antecedent'])[:25]
                con_name = product_names.get(rule['consequent'], rule['consequent'])[:25]
                
                print(f"\n{i+1}. {ant_name} → {con_name}")
                print(f"   Support: {rule['support']:.2%}")
                print(f"   Confidence: {rule['confidence']:.2%}")
                print(f"   Lift: {rule['lift']:.2f}")
                
                if rule['lift'] > 1.5:
                    strength = "Strong"
                elif rule['lift'] > 1.2:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                print(f"   Association strength: {strength}")
        else:
            print("   No strong association rules found with current thresholds.")
        
        # Category-based associations
        print(f"\n🏷️ Category Association Analysis:")
        
        # Add category info to transactions
        transactions_with_category = self.transactions_df.merge(
            self.products_df[['product_id', 'category']], on='product_id'
        )
        
        # Find category pairs in same baskets
        from itertools import combinations
        category_baskets = transactions_with_category.groupby([
            'customer_id', 'transaction_date'
        ])['category'].apply(lambda x: list(set(x))).reset_index()
        
        category_pairs = {}
        for basket in category_baskets['category']:
            if len(basket) >= 2:
                for cat1, cat2 in combinations(basket, 2):
                    pair = tuple(sorted([str(cat1), str(cat2)]))  # Patch: ensure both are str
                    category_pairs[pair] = category_pairs.get(pair, 0) + 1
        
        # Sort category pairs
        sorted_pairs = sorted(category_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   Top category combinations:")
        for i, ((cat1, cat2), count) in enumerate(sorted_pairs[:5]):
            frequency = count / len(category_baskets)
            print(f"   {i+1}. {cat1} + {cat2}: {count} times ({frequency:.1%})")
        
        # Recommendation function
        def get_recommendations(product_id, top_n=3):
            recommendations = []
            for rule in rules:
                if rule['antecedent'] == product_id:
                    rec_name = product_names.get(rule['consequent'], rule['consequent'])
                    recommendations.append((rec_name, rule['confidence'], rule['lift']))
            
            return sorted(recommendations, key=lambda x: x[2], reverse=True)[:top_n]
        
        # Example recommendation
        if rules:
            sample_product = rules[0]['antecedent']
            sample_name = product_names.get(sample_product, sample_product)[:30]
            recommendations = get_recommendations(sample_product)
            
            print(f"\n🔮 Recommendation Example:")
            print(f"   Customer viewing: {sample_name}")
            print(f"   Recommended products:")
            
            for i, (rec_name, confidence, lift) in enumerate(recommendations):
                print(f"   {i+1}. {rec_name[:30]} (confidence: {confidence:.1%})")
        
        # Save rules
        self.models['association_rules'] = rules
    
    def save_models(self):
        """Save all trained models"""
        with open('marketplace_models.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers
            }, f)
        print("💾 Models saved to marketplace_models.pkl")
    
    def load_models(self):
        """Load saved models"""
        try:
            with open('marketplace_models.pkl', 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.scalers = data['scalers']
            print("📁 Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("❌ No saved models found.")
            return False

class MarketplaceCLI:
    """Command Line Interface for the marketplace"""
    
    def __init__(self):
        self.ml_system = MarketplaceML()
        self.running = True
        self.ml_system.load_data()
    
    def display_header(self):
        """Display application header"""
        print("\n" + "="*60)
        print("🛒 SMARTMARKET CLI - ML Powered Marketplace")
        print("="*60)
      
    
    def display_menu(self):
        """Display main menu"""
        print("\n📋 MAIN MENU")
        print("-" * 20)
        print("1. 📊 Load/Generate Data")
        print("2. 📈 Regression Analysis (Price & Sales Prediction)")
        print("3. 🏷️ Classification Analysis (Satisfaction Prediction)")
        print("4. 👥 Clustering Analysis (Customer Segmentation)")
        print("5. 🛒 Association Rules (Product Recommendations)")
        print("6. 🔄 Run All Analyses")
        print("7. 💾 Save Models")
        print("8. 📁 Load Models")
        print("9. 📊 Show Data Summary")
        print("10. ❓ Ask a Question (Data Query)")
        print("0. 🚪 Exit")
        print("-" * 20)
    
    def show_data_summary(self):
        """Show summary of loaded data"""
        if self.ml_system.products_df is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n📊 DATA SUMMARY")
        print("=" * 30)
        
        # Products summary
        print(f"📦 Products: {len(self.ml_system.products_df)}")
        print("   Categories:", ", ".join(str(cat) for cat in self.ml_system.products_df['category'].unique()))
        print(f"   Price range: ${self.ml_system.products_df['price'].min():.2f} - ${self.ml_system.products_df['price'].max():.2f}")
        
        # Customers summary
        print(f"\n👥 Customers: {len(self.ml_system.customers_df)}")
        print("   Age groups:", ", ".join(str(ag) for ag in self.ml_system.customers_df['age_group'].unique()))
        print(f"   Prime members: {self.ml_system.customers_df['prime_member'].sum()}")
        
        # Transactions summary
        print(f"\n🛒 Transactions: {len(self.ml_system.transactions_df)}")
        print(f"   Total revenue: ${self.ml_system.transactions_df['total_amount'].sum():.2f}")
        print(f"   Average order value: ${self.ml_system.transactions_df['total_amount'].mean():.2f}")
        
        # Satisfaction distribution
        satisfaction_dist = self.ml_system.transactions_df['satisfaction_category'].value_counts()
        print(f"\n😊 Satisfaction Distribution:")
        for category, count in satisfaction_dist.items():
            percentage = count / len(self.ml_system.transactions_df) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
    
    def run_all_analyses(self):
        """Run all ML analyses"""
        if self.ml_system.products_df is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n🚀 RUNNING ALL ML ANALYSES")
        print("=" * 40)
        
        try:
            self.ml_system.regression_analysis()
            input("\nPress Enter to continue to Classification...")
            
            self.ml_system.classification_analysis()
            input("\nPress Enter to continue to Clustering...")
            
            self.ml_system.clustering_analysis()
            input("\nPress Enter to continue to Association Rules...")
            
            self.ml_system.association_rules_analysis()
            
            print("\n🎉 All analyses completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
    
    def ask_question(self):
        import os
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            api_key = input('Enter your OpenAI API key: ')
        llm = OpenAI(api_token=api_key)
        print("\nAsk any question about your marketplace data!")
        print("Examples: 'sales in March', 'revenue in 2018', 'top product', 'highest margin', 'top customer', 'orders in Q2 2017', 'plot sales trend'...")
        question = input("Type your question: ")
        try:
            merged_df = self.ml_system.transactions_df.merge(
                self.ml_system.products_df, on='product_id', how='left'
            ).merge(
                self.ml_system.customers_df, on='customer_id', how='left'
            )
            # Keep only relevant columns
            columns_to_keep = [
                'transaction_id', 'product_id', 'customer_id', 'category', 'total_amount', 'month', 'transaction_date',
                'unit_price', 'quantity', 'satisfaction_score', 'satisfaction_category', 'delivery_time', 'brand', 'price', 'rating', 'age_group', 'location', 'income_level', 'tech_savvy', 'prime_member', 'avg_order_value', 'loyalty_score'
            ]
            merged_df = merged_df[[col for col in columns_to_keep if col in merged_df.columns]]
            # No sampling: use the whole DataFrame
            sdf = SmartDataframe(merged_df, config={"llm": llm})
            result = sdf.chat(question)
            print(result)
        except Exception as e:
            print(f"Error using SmartDataframe: {e}")
        input("\nPress Enter to return to the menu...")

    
    def run(self):
        """Main application loop"""
        self.display_header()
        
        while self.running:
            try:
                self.display_menu()
                choice = input("\n🎯 Enter your choice (0-10): ").strip()
                
                if choice == '1':
                    self.ml_system.load_data()
                
                elif choice == '2':
                    if self.ml_system.products_df is None:
                        print("❌ Please load data first (option 1)")
                    else:
                        self.ml_system.regression_analysis()
                
                elif choice == '3':
                    if self.ml_system.products_df is None:
                        print("❌ Please load data first (option 1)")
                    else:
                        self.ml_system.classification_analysis()
                
                elif choice == '4':
                    if self.ml_system.products_df is None:
                        print("❌ Please load data first (option 1)")
                    else:
                        self.ml_system.clustering_analysis()
                
                elif choice == '5':
                    if self.ml_system.products_df is None:
                        print("❌ Please load data first (option 1)")
                    else:
                        self.ml_system.association_rules_analysis()
                
                elif choice == '6':
                    self.run_all_analyses()
                
                elif choice == '7':
                    if self.ml_system.models:
                        self.ml_system.save_models()
                    else:
                        print("❌ No models to save. Run analyses first.")
                
                elif choice == '8':
                    self.ml_system.load_models()
                
                elif choice == '9':
                    self.show_data_summary()
                
                elif choice == '10':
                    if self.ml_system.products_df is None:
                        print("❌ Please load data first (option 1)")
                    else:
                        self.ask_question()
                
                
                
                elif choice == '0':
                    print("\n👋 Thank you for using SmartMarket CLI!")
                    self.running = False
                
                else:
                    print("❌ Invalid choice. Please try again.")
                
                if self.running and choice != '0':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                input("Press Enter to continue...")

def main():
    """Main function"""
    try:
        cli = MarketplaceCLI()
        cli.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()