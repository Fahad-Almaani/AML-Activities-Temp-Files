import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Number of clusters for K-Means
N_CLUSTERS = 3

# Load the dataset
df = pd.read_csv('HR.csv')

# Select columns to use in clustering
NUMERIC_FEATURES = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_montly_hours', 'time_spend_company', 'sales', 'salary']

# Encode 'sales' and 'salary' columns using Label Encoding
label_encoder = LabelEncoder()

# Encode 'sales' and 'salary' columns
df['sales'] = label_encoder.fit_transform(df['sales'])
df['salary'] = label_encoder.fit_transform(df['salary'])

# One-hot encoding for other categorical features (if any)
# If there are other categorical columns you want to encode (besides 'sales' and 'salary'), you can apply get_dummies here.
# For now, we don't need to apply it for 'sales' and 'salary' as we already encoded them numerically.

# Standardize numeric columns
scaler = StandardScaler()
df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[NUMERIC_FEATURES])

# Visualize clustering results (e.g., satisfaction_level vs. average_montly_hours)
plt.figure(figsize=(8, 6))
for i in range(N_CLUSTERS):
    plt.scatter(df[df['Cluster'] == i]['satisfaction_level'], 
                df[df['Cluster'] == i]['average_montly_hours'], 
                label=f'Cluster {i+1}')
plt.xlabel('Satisfaction Level')
plt.ylabel('Average Monthly Hours')
plt.legend()
plt.title('K-Means Clustering Results')
plt.show()

# Summarize clusters
cluster_summary = df.groupby('Cluster').mean()
cluster_summary['Count'] = df['Cluster'].value_counts()
cluster_summary = cluster_summary.sort_values(by='Count', ascending=False)
print(cluster_summary)
