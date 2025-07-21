#!/usr/bin/env python3
"""
📊 SmartMarket Data Viewer
Simple script to explore the generated marketplace datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_datasets():
    """Load all CSV datasets"""
    datasets = {}
    
    files = {
        'products': 'data/products.csv',
        'customers': 'data/customers.csv', 
        'transactions': 'data/transactions.csv'
    }
    
    for name, filename in files.items():
        if os.path.exists(filename):
            datasets[name] = pd.read_csv(filename)
            print(f"✅ Loaded {filename}: {len(datasets[name])} records")
        else:
            print(f"❌ {filename} not found. Run the main application first.")
            return None
    
    return datasets

def explore_products(df):
    """Explore products dataset"""
    print("\n📦 PRODUCTS ANALYSIS")
    print("=" * 40)
    
    print(f"Total products: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Brands: {df['brand'].nunique()}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"Average price: ${df['price'].mean():.2f}")
    
    print("\n📊 Category Distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = count / len(df) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    print("\n💰 Price by Category:")
    price_by_category = df.groupby('category')['price'].agg(['mean', 'min', 'max'])
    for category in price_by_category.index:
        mean_price = price_by_category.loc[category, 'mean']
        min_price = price_by_category.loc[category, 'min']
        max_price = price_by_category.loc[category, 'max']
        print(f"   {category}: ${mean_price:.2f} (${min_price:.2f} - ${max_price:.2f})")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Category distribution
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Product Distribution by Category')
    
    # Price distribution
    axes[0, 1].hist(df['price'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Price ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Price Distribution')
    
    # Price by category boxplot
    df.boxplot(column='price', by='category', ax=axes[1, 0])
    axes[1, 0].set_title('Price by Category')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Rating vs Price scatter
    axes[1, 1].scatter(df['rating'], df['price'], alpha=0.6)
    axes[1, 1].set_xlabel('Rating')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].set_title('Rating vs Price')
    
    plt.suptitle('Products Dataset Analysis', y=1.02)
    plt.tight_layout()
    plt.show()

def explore_customers(df):
    """Explore customers dataset"""
    print("\n👥 CUSTOMERS ANALYSIS")
    print("=" * 40)
    
    print(f"Total customers: {len(df)}")
    print(f"Prime members: {df['prime_member'].sum()} ({df['prime_member'].mean()*100:.1f}%)")
    print(f"Tech savvy: {df['tech_savvy'].sum()} ({df['tech_savvy'].mean()*100:.1f}%)")
    
    print("\n📊 Demographics:")
    
    print("Age Groups:")
    age_counts = df['age_group'].value_counts()
    for age_group, count in age_counts.items():
        percentage = count / len(df) * 100
        print(f"   {age_group}: {count} ({percentage:.1f}%)")
    
    print("\nLocations:")
    location_counts = df['location'].value_counts()
    for location, count in location_counts.items():
        percentage = count / len(df) * 100
        print(f"   {location}: {count} ({percentage:.1f}%)")
    
    print("\nIncome Levels:")
    income_counts = df['income_level'].value_counts()
    for income, count in income_counts.items():
        percentage = count / len(df) * 100
        print(f"   {income}: {count} ({percentage:.1f}%)")
    
    print(f"\n💰 Average Order Value: ${df['avg_order_value'].mean():.2f}")
    print(f"Loyalty Score: {df['loyalty_score'].mean():.1f}/100")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Age group distribution
    age_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Age Group Distribution')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Location distribution
    location_counts.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Location Distribution')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Income distribution
    income_counts.plot(kind='bar', ax=axes[0, 2], color='lightcoral')
    axes[0, 2].set_title('Income Level Distribution')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Average order value distribution
    axes[1, 0].hist(df['avg_order_value'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Average Order Value ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Average Order Value Distribution')
    
    # Loyalty score distribution
    axes[1, 1].hist(df['loyalty_score'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Loyalty Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Loyalty Score Distribution')
    
    # Prime vs non-prime order values
    prime_data = df[df['prime_member'] == 1]['avg_order_value']
    non_prime_data = df[df['prime_member'] == 0]['avg_order_value']
    axes[1, 2].hist([non_prime_data, prime_data], bins=20, alpha=0.7, 
                   label=['Non-Prime', 'Prime'], color=['red', 'blue'])
    axes[1, 2].set_xlabel('Average Order Value ($)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Order Value: Prime vs Non-Prime')
    axes[1, 2].legend()
    
    plt.suptitle('Customers Dataset Analysis', y=1.02)
    plt.tight_layout()
    plt.show()

def explore_transactions(df, products_df, customers_df):
    """Explore transactions dataset"""
    print("\n🛒 TRANSACTIONS ANALYSIS")
    print("=" * 40)
    
    print(f"Total transactions: {len(df)}")
    print(f"Total revenue: ${df['total_amount'].sum():,.2f}")
    print(f"Average order value: ${df['total_amount'].mean():.2f}")
    print(f"Average delivery time: {df['delivery_time'].mean():.1f} days")
    print(f"Average satisfaction: {df['satisfaction_score'].mean():.1f}/100")
    
    print("\n😊 Satisfaction Distribution:")
    satisfaction_counts = df['satisfaction_category'].value_counts()
    for satisfaction, count in satisfaction_counts.items():
        percentage = count / len(df) * 100
        print(f"   {satisfaction}: {count} ({percentage:.1f}%)")
    
    # Merge with products for category analysis
    merged_df = df.merge(products_df[['product_id', 'category']], on='product_id')
    
    print("\n📊 Sales by Category:")
    category_sales = merged_df.groupby('category')['total_amount'].agg(['sum', 'count', 'mean'])
    category_sales = category_sales.sort_values('sum', ascending=False)
    
    for category in category_sales.index:
        total_sales = category_sales.loc[category, 'sum']
        num_transactions = category_sales.loc[category, 'count']
        avg_value = category_sales.loc[category, 'mean']
        print(f"   {category}: ${total_sales:,.2f} ({num_transactions} orders, ${avg_value:.2f} avg)")
    
    # Monthly trends
    print("\n📅 Monthly Sales Trends:")
    monthly_sales = df.groupby('month')['total_amount'].agg(['sum', 'count'])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month in sorted(monthly_sales.index):
        total_sales = monthly_sales.loc[month, 'sum']
        num_orders = monthly_sales.loc[month, 'count']
        month_name = month_names[month-1]
        print(f"   {month_name}: ${total_sales:,.2f} ({num_orders} orders)")
    
    # Visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Revenue by category
    category_revenue = merged_df.groupby('category')['total_amount'].sum().sort_values(ascending=False)
    category_revenue.plot(kind='bar', ax=axes[0, 0], color='gold')
    axes[0, 0].set_title('Revenue by Category')
    axes[0, 0].set_ylabel('Revenue ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Order value distribution
    axes[0, 1].hist(df['total_amount'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 1].set_xlabel('Order Value ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Order Value Distribution')
    
    # Delivery time distribution
    axes[1, 0].hist(df['delivery_time'], bins=range(1, int(df['delivery_time'].max())+2), 
                   alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Delivery Time (days)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Delivery Time Distribution')
    
    # Satisfaction score distribution
    axes[1, 1].hist(df['satisfaction_score'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 1].set_xlabel('Satisfaction Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Satisfaction Score Distribution')
    
    # Monthly sales trend
    monthly_revenue = df.groupby('month')['total_amount'].sum()
    axes[2, 0].plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2, markersize=8)
    axes[2, 0].set_xlabel('Month')
    axes[2, 0].set_ylabel('Revenue ($)')
    axes[2, 0].set_title('Monthly Revenue Trend')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Satisfaction by delivery time
    satisfaction_delivery = df.groupby('delivery_time')['satisfaction_score'].mean()
    axes[2, 1].plot(satisfaction_delivery.index, satisfaction_delivery.values, 
                   marker='o', linewidth=2, markersize=6, color='red')
    axes[2, 1].set_xlabel('Delivery Time (days)')
    axes[2, 1].set_ylabel('Average Satisfaction Score')
    axes[2, 1].set_title('Satisfaction vs Delivery Time')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Transactions Dataset Analysis', y=1.02)
    plt.tight_layout()
    plt.show()

def cross_dataset_analysis(datasets):
    """Analyze relationships across datasets"""
    print("\n🔄 CROSS-DATASET ANALYSIS")
    print("=" * 40)
    
    products_df = datasets['products']
    customers_df = datasets['customers']
    transactions_df = datasets['transactions']
    
    # Merge all datasets
    full_df = transactions_df.merge(products_df, on='product_id')
    full_df = full_df.merge(customers_df, on='customer_id')
    
    print("🎯 Key Business Insights:")
    
    # Prime vs non-prime analysis
    prime_stats = full_df[full_df['prime_member'] == 1]['total_amount'].agg(['mean', 'sum', 'count'])
    non_prime_stats = full_df[full_df['prime_member'] == 0]['total_amount'].agg(['mean', 'sum', 'count'])
    
    print(f"\n💎 Prime Members:")
    print(f"   Average order value: ${prime_stats['mean']:.2f}")
    print(f"   Total revenue: ${prime_stats['sum']:,.2f}")
    print(f"   Number of orders: {prime_stats['count']}")
    
    print(f"\n👤 Non-Prime Members:")
    print(f"   Average order value: ${non_prime_stats['mean']:.2f}")
    print(f"   Total revenue: ${non_prime_stats['sum']:,.2f}")
    print(f"   Number of orders: {non_prime_stats['count']}")
    
    # Category preferences by demographics
    print(f"\n🛍️ Category Preferences by Age Group:")
    age_category = full_df.groupby(['age_group', 'category'])['total_amount'].sum().unstack(fill_value=0)
    
    for age_group in age_category.index:
        top_category = age_category.loc[age_group].idxmax()
        top_amount = age_category.loc[age_group].max()
        print(f"   {age_group}: Prefers {top_category} (${top_amount:,.2f})")
    
    # Income vs spending correlation
    income_spending = full_df.groupby(['customer_id', 'income_level'])['total_amount'].sum().reset_index()
    income_avg = income_spending.groupby('income_level')['total_amount'].mean()
    
    print(f"\n💰 Spending by Income Level:")
    for income_level in ['Low', 'Medium', 'High']:
        if income_level in income_avg.index:
            avg_spending = income_avg[income_level]
            print(f"   {income_level} income: ${avg_spending:.2f} average per customer")
    
    # Best performing products
    product_performance = full_df.groupby('product_id').agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'satisfaction_score': 'mean'
    }).sort_values('total_amount', ascending=False)
    
    print(f"\n🏆 Top 5 Best-Selling Products:")
    for i, (product_id, row) in enumerate(product_performance.head().iterrows()):
        product_name = products_df[products_df['product_id'] == product_id]['name'].iloc[0]
        revenue = row['total_amount']
        units = row['quantity']
        satisfaction = row['satisfaction_score']
        print(f"   {i+1}. {product_name[:40]}")
        print(f"      Revenue: ${revenue:.2f}, Units: {units}, Satisfaction: {satisfaction:.1f}")

def generate_summary_report(datasets):
    """Generate a comprehensive summary report"""
    print("\n📋 SUMMARY REPORT")
    print("=" * 50)
    
    products_df = datasets['products']
    customers_df = datasets['customers']
    transactions_df = datasets['transactions']
    
    # Overall metrics
    total_revenue = transactions_df['total_amount'].sum()
    total_customers = customers_df['customer_id'].nunique()
    total_products = products_df['product_id'].nunique()
    total_orders = len(transactions_df)
    avg_order_value = transactions_df['total_amount'].mean()
    
    print(f"🏪 MARKETPLACE OVERVIEW")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Total Customers: {total_customers:,}")
    print(f"   Total Products: {total_products:,}")
    print(f"   Total Orders: {total_orders:,}")
    print(f"   Average Order Value: ${avg_order_value:.2f}")
    
    # Customer metrics
    prime_percentage = customers_df['prime_member'].mean() * 100
    tech_savvy_percentage = customers_df['tech_savvy'].mean() * 100
    avg_loyalty = customers_df['loyalty_score'].mean()
    
    print(f"\n👥 CUSTOMER METRICS")
    print(f"   Prime Members: {prime_percentage:.1f}%")
    print(f"   Tech Savvy: {tech_savvy_percentage:.1f}%")
    print(f"   Average Loyalty Score: {avg_loyalty:.1f}/100")
    
    # Operational metrics
    avg_delivery_time = transactions_df['delivery_time'].mean()
    avg_satisfaction = transactions_df['satisfaction_score'].mean()
    high_satisfaction_pct = (transactions_df['satisfaction_category'] == 'High').mean() * 100
    
    print(f"\n📦 OPERATIONAL METRICS")
    print(f"   Average Delivery Time: {avg_delivery_time:.1f} days")
    print(f"   Average Satisfaction: {avg_satisfaction:.1f}/100")
    print(f"   High Satisfaction Rate: {high_satisfaction_pct:.1f}%")
    
    # Product metrics
    avg_product_price = products_df['price'].mean()
    avg_product_rating = products_df['rating'].mean()
    prime_eligible_pct = products_df['is_prime'].mean() * 100
    
    print(f"\n📦 PRODUCT METRICS")
    print(f"   Average Product Price: ${avg_product_price:.2f}")
    print(f"   Average Product Rating: {avg_product_rating:.1f}/5.0")
    print(f"   Prime Eligible Products: {prime_eligible_pct:.1f}%")
    
    print(f"\n✨ DATA QUALITY")
    print(f"   Products dataset: {len(products_df)} records, {products_df.isnull().sum().sum()} missing values")
    print(f"   Customers dataset: {len(customers_df)} records, {customers_df.isnull().sum().sum()} missing values")
    print(f"   Transactions dataset: {len(transactions_df)} records, {transactions_df.isnull().sum().sum()} missing values")

def main():
    """Main data exploration function"""
    print("📊 SMARTMARKET DATA EXPLORER")
    print("=" * 50)
    print("This tool helps you explore the generated marketplace datasets")
    print()
    
    # Load datasets
    datasets = load_datasets()
    if datasets is None:
        return
    
    print("\n🎯 What would you like to explore?")
    print("1. 📦 Products Analysis")
    print("2. 👥 Customers Analysis") 
    print("3. 🛒 Transactions Analysis")
    print("4. 🔄 Cross-Dataset Analysis")
    print("5. 📋 Complete Summary Report")
    print("6. 🎨 All Visualizations")
    print("0. Exit")
    
    while True:
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            explore_products(datasets['products'])
        elif choice == '2':
            explore_customers(datasets['customers'])
        elif choice == '3':
            explore_transactions(datasets['transactions'], 
                               datasets['products'], datasets['customers'])
        elif choice == '4':
            cross_dataset_analysis(datasets)
        elif choice == '5':
            generate_summary_report(datasets)
        elif choice == '6':
            print("\n🎨 Generating all visualizations...")
            explore_products(datasets['products'])
            explore_customers(datasets['customers'])
            explore_transactions(datasets['transactions'], 
                               datasets['products'], datasets['customers'])
        elif choice == '0':
            print("\n👋 Thank you for exploring the data!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        if choice != '0':
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()