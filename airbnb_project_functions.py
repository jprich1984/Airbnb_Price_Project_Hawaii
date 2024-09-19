import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.stats import ttest_ind
from scipy import stats
import re
import swifter
import random
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import csv
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def plot_listings_against_column(df, rating_columns, chosen_column):
    """
    Plots each listing_id against the chosen column for each rating column.

    Parameters:
    - df: DataFrame containing the data
    - rating_columns: List of columns representing ratings
    - chosen_column: The column to plot ratings against
    """
    # Set the style of seaborn
    sns.set(style="whitegrid")
    
    # Create a figure with subplots
    n = len(rating_columns)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 2 * n))
    
    # Loop through each rating column and create a plot
    for i, rating in enumerate(rating_columns):
        ax = axes[i] if n > 1 else axes  # Handle case where there's only one plot
        
        # Scatter plot of listing_id against the chosen column
        sns.scatterplot(data=df, x='listing_id', y=chosen_column, ax=ax, label=chosen_column, alpha=0.5)
        
        # Overlay the rating column
        sns.scatterplot(data=df, x='listing_id', y=rating, ax=ax, label=rating, alpha=0.5)
        
        ax.set_title(f'{chosen_column} and {rating} vs Listing ID')
        ax.set_xlabel('Listing ID')
        ax.set_ylabel('Value')
        ax.axhline(y=0, color='gray', linestyle='--')  # Optional: add a horizontal line at y=0 for reference
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
def format_number(df,x):
    desc_stats = df['price'].describe()
    if pd.isna(x):
        return "N/A"
    elif abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    elif abs(x) >= 1e3:
        return f"${x/1e3:.2f}K"
    else:
        return f"${x:.2f}"




def plot_column_against_price(df,column):
    gf=df.copy()
    grouped=gf.groupby(column).agg({'price':'mean'}).reset_index()
    fig, ax=plt.subplots()
    ax.scatter(column,'price',data=grouped)
    ax.set_xlabel(column)
    ax.set_ylabel('price')
    fig.suptitle(f'{column} vs. Price')
    plt.tight_layout()
    plt.show()

def impute_columns(row):
    if pd.isna(row['bedrooms']):
        # First, check for 'bedrooms' or 'Bedrooms'
        pattern_bedrooms = r'(\d+)\s*[Bb]edrooms?'
        match_bedrooms = re.search(pattern_bedrooms, row['name'])
        
        if match_bedrooms:
            bedrooms = int(match_bedrooms.group(1))
        else:
            # If 'bedrooms' not found, check for 'studio' or 'Studio'
            pattern_studio = r'\b[Ss]tudio\b'
            match_studio = re.search(pattern_studio, row['name'])
            
            if match_studio:
                bedrooms = 0  # or 1, depending on how you want to count studios
            else:
                bedrooms = np.nan
    else:
        bedrooms = row['bedrooms']
    
    # Handle bathrooms (unchanged from your original function)
    if pd.isna(row['bathrooms']):
        if pd.isna(row['bathrooms_text']):
            bathrooms = np.nan
        
        elif 'half-bath' in row['bathrooms_text'].lower():
            bathrooms = 0.5
        else:
            try:
                bathrooms = float(row['bathrooms_text'].split(" ")[0])
            except (ValueError, IndexError):
                bathrooms = np.nan
    else:
        bathrooms = row['bathrooms']
    
    if pd.isna(row['bathrooms_text']):
        shared = np.nan
    else:
        if 'shared' in row['bathrooms_text'].lower():
            shared = 1
        else:
            shared = 0
    return pd.Series([bedrooms, bathrooms,shared], index=['bedrooms', 'bathrooms','isShared_bathrooms'])

def read_zipped_csv(file_path):
    # Initialize an empty list to hold the chunks
    chunks = []

    # Count total number of rows and read data in chunks
    total_rows = 0
    for chunk in pd.read_csv(file_path, compression='gzip', chunksize=10000):
        total_rows += len(chunk)
        chunks.append(chunk)

    print(f"Total number of rows: {total_rows}")

    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)
    return df
def get_outliers(data,col,factor):
    quarts=data[col].quantile([0.25,0.75])
    iqr=quarts.loc[0.75]-quarts.loc[0.25]
    upper_boundary=quarts.loc[0.75]+iqr*factor
    lower_boundary=quarts.loc[0.25]-iqr*factor
    return data[(data[col]<lower_boundary)|(data[col]>upper_boundary)]

def check_normality(data, variable_name="Data"):
    n = len(data)
    results = {}
    
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    results["Shapiro-Wilk"] = {
        "statistic": shapiro_stat,
        "p-value": shapiro_p,
        "normal": shapiro_p > 0.05
    }
    if n > 5000:
        results["Shapiro-Wilk"]["note"] = "Large sample size: p-value may be overly sensitive to small deviations."
    
    # Anderson-Darling Test
    anderson_result = stats.anderson(data, dist='norm')
    results["Anderson-Darling"] = {
        "statistic": anderson_result.statistic,
        "critical_values": anderson_result.critical_values,
        "significance_level": anderson_result.significance_level,
        "normal": anderson_result.statistic < anderson_result.critical_values[2]  # Using 5% significance level
    }
    
    # Visual inspection
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {variable_name}")
    
    plt.subplot(122)
    plt.hist(data, bins=50, density=True, alpha=0.7)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f"Histogram with Normal Curve: {variable_name}")
    plt.show()
    
    # Print results
    print(f"\nNormality Test Results for {variable_name}:")
    for test, result in results.items():
        print(f"{test}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        if "normal" in result:
            print(f"  Conclusion: The data {'is' if result['normal'] else 'is not'} normally distributed according to this test.")
    
    return results
def remove_outliers(data, column, method='iqr'):
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data[column]))
        return data[z_scores < 3]
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
def plot_aggregated_column_vs_date(df, y_column):
    """
    Groups the DataFrame by 'date', calculates the mean of the specified column,
    and plots the average values against the date.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    y_column (str): The name of the column to plot on the y-axis.

    Returns:
    None
    """
    # Ensure 'date' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Group by 'date' and calculate the mean of the specified column
    aggregated_df = df.groupby('date')[y_column].mean().reset_index()

    # Plot the aggregated data
    plt.figure(figsize=(12, 6))
    plt.plot(aggregated_df['date'], aggregated_df[y_column], label=f'Average {y_column}')
    plt.title(f'Average {y_column} vs Date')
    plt.xlabel('Date')
    plt.ylabel(f'Average {y_column}')
    plt.legend()
    plt.grid(True)
    plt.show()

def look_at_listing_pool_amenities(url):
    specific_listing=df_pool_mentioned[df_pool_mentioned['listing_url']==url]['amenities'].iloc[0]
    print("Pool amenities mentioned")
    for amenity in specific_listing:
        pattern1 = r"private pool"
        pattern2 = r"private hot tub"   
        if isinstance(amenity, str):
            if re.search(pattern1, amenity, re.IGNORECASE) or re.search(pattern2, amenity, re.IGNORECASE):     
                print(f"- {amenity}")
def train_test_split_airbnb(df,train_size):
    listings=df['listing_id'].unique()
    indices = list(range(len(listings)))
    train_indices=random.sample(indices, int(train_size*len(indices)))
    test_indices=[ ind for ind in indices if ind not in train_indices]
    train_listings=[listings[ind] for ind in train_indices]
    test_listings=[listings[ind] for ind in test_indices]
    train=df[df['listing_id'].isin(train_listings)]
    test=df[df['listing_id'].isin(test_listings)]
    return train,test
def custom_cross_validation(df,seed, n_splits=5):
    # Get unique listing IDs
    listings = df['listing_id'].unique()
    n_listings = len(listings)
    random.seed(seed)
    # Shuffle the listings
    random.shuffle(listings)
    
    # Calculate the size of each fold
    fold_size = n_listings // n_splits
    
    # Create folds as a list of lists
    folds = [list(listings[i * fold_size:(i + 1) * fold_size]) for i in range(n_splits)]
    
    # If there are leftover listings, distribute them into the folds
    leftover = listings[n_splits * fold_size:]
    for i, listing in enumerate(leftover):
        folds[i].append(listing)
    
    # Generate train-test splits
    for i in range(n_splits):
        # Validation fold
        val_listings = folds[i]
        
        # Training folds (all other folds)
        train_listings = [listing for j in range(n_splits) if j != i for listing in folds[j]]
        
        # Create train and validation DataFrames
        train = df[df['listing_id'].isin(train_listings)]
        val = df[df['listing_id'].isin(val_listings)]
        
        yield train, val


def permutation_importance_cv(model, X, y, feature, n_repeats=12):
    baseline_mse = mean_squared_error(y, model.predict(X))
    importances = []
    for _ in range(n_repeats):
        X_permuted = X.copy()
        X_permuted[feature] = np.random.permutation(X_permuted[feature])
        permuted_mse = mean_squared_error(y, model.predict(X_permuted))
        importances.append(permuted_mse - baseline_mse)
    return np.mean(importances)

def cross_validated_feature_importance(df, model_class, model_params, n_splits=5):
    feature_importances = []
    non_features=['price','inResort','number_of_reviews','geometry','week','listing_id','log_price','listing_url','missing_rating','price_per_accommodation','price_per_bedroom','price_per_review','property_type','geometry','anomaly_score','price_per_accommodation','price_per_bedroom','price_per_review','amenities_length']
    
    features=[col for col in df.columns if col not in non_features]
    seed=2
    for train, val in custom_cross_validation(df, seed,n_splits=n_splits):
        X_train = train[features]
        y_train = train['price']
        X_val = val[features]
        y_val = val['price']
        
        # Train the model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Calculate feature importance for this fold
        fold_importances = []
        for feature in X_val.columns:
            importance = permutation_importance_cv(model, X_val, y_val, feature)
            fold_importances.append({'feature': feature, 'importance': importance})
        
        feature_importances.append(pd.DataFrame(fold_importances))
    
    # Aggregate importances across folds
    all_importances = pd.concat(feature_importances)
    mean_importances = all_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
    std_importances = all_importances.groupby('feature')['importance'].std()
    
    return mean_importances, std_importances

def specific_listing(df):
    listings=df['listing_id'].unique()
    return df[df['listing_id']==random.choice(listings)]

def detect_outliers(X, contamination=0.01):
    """
    Detect outliers using Isolation Forest.
    
    Parameters:
    X (DataFrame): Features to use for outlier detection
    contamination (float): The proportion of outliers in the data set
    
    Returns:
    numpy array: Boolean mask where True indicates an outlier
    """
    # Create and fit the Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X)
    
    # Predict anomalies (-1 for outliers, 1 for inliers)
    y_pred = clf.predict(X)
    
    # Convert to boolean mask (True for outliers)
    outlier_mask = y_pred == -1
    
    return outlier_mask



def detect_and_return_outliers(df, features, contamination=0.01):
    """
    Detect outliers using Isolation Forest and return them.
    
    Parameters:
    df (DataFrame): The full DataFrame
    features (list): List of feature column names to use for outlier detection
    contamination (float): The proportion of outliers in the data set
    
    Returns:
    DataFrame: The outlier rows from the original DataFrame
    """
    # Extract features for outlier detection
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and fit the Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X_scaled)
    
    # Predict anomalies (-1 for outliers, 1 for inliers)
    y_pred = clf.predict(X_scaled)
    
    # Create a boolean mask (True for outliers)
    outlier_mask = y_pred == -1
    
    # Return the outlier rows
    outliers = df[outlier_mask]
    
    return outliers

def best_eps(X,num_neighbors,plot=False):
    nbrs = NearestNeighbors(n_neighbors= num_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    opt_eps_minpts=np.empty((0,3))
    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows= 1, ncols= 2, figsize= (12, 6))
    for i in range(1,num_neighbors):
        dist_i=np.sort(distances[:,i])
        min_dist=dist_i[0]
        max_dist=dist_i[-1]
        values=np.linspace(min_dist,max_dist, len(dist_i))
        distA = np.concatenate([np.arange(1, dist_i.shape[0] + 1) / dist_i.shape[0],dist_i]).reshape(2, -1).T
        distB = np.concatenate([np.arange(1, dist_i.shape[0] + 1) / len(dist_i), values]).reshape(2, -1).T
        all2all = cdist(XA = distA, XB= distB)
        min_distance_ind=all2all.min(axis= 1).argmax()
        opt_eps_minpts = np.vstack([opt_eps_minpts, np.array([i, min_distance_ind, dist_i[min_distance_ind]])])
        if plot:
            ax1.plot(dist_i, label= f'k={i}')
            ax1.scatter(x= min_distance_ind, y= dist_i[min_distance_ind])
            ax2.plot(dist_i, label= f'k={i}')
            ax2.scatter(x= min_distance_ind, y= dist_i[min_distance_ind])
    eps_delta = opt_eps_minpts[:, 2].max() - opt_eps_minpts[:, 2].min()
    epsilon=np.mean([x[2] for x in opt_eps_minpts])
    return epsilon

