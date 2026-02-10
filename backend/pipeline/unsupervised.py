import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class AutoEDA:
    """
    Automatic Exploratory Data Analysis for Automobile Datasets
    Designed for unsupervised learning pipelines
    """
    
    def __init__(self, df, output_dir=None):
        """
        Initialize AutoEDA
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed automobile dataset
        output_dir : str, optional
            Directory to save plots (if None, plots are only displayed)
        """
        self.df = df.copy()
        self.output_dir = output_dir
        
        # Identify column types
        self.numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print("="*80)
        print("AUTO EDA INITIALIZED")
        print("="*80)
        print(f"Numerical columns: {len(self.numerical_cols)}")
        print(f"Categorical columns: {len(self.categorical_cols)}")
        print("="*80)
    
    
    def generate_full_report(self):
        """
        Generate complete EDA report with all analyses
        """
        print("\n🔍 Starting Automatic EDA...\n")
        
        self.dataset_overview()
        self.distribution_analysis()
        self.correlation_analysis()
        self.outlier_detection()
        self.categorical_analysis()
        self.variance_analysis()
        self.feature_relationships()
        self.pca_analysis()
        
        print("\n✅ EDA Complete!")
        print("="*80)
    
    
    def dataset_overview(self):
        """
        Display basic dataset information
        """
        print("\n" + "="*80)
        print("📊 DATASET OVERVIEW")
        print("="*80)
        
        print(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n--- Data Types ---")
        print(self.df.dtypes.value_counts())
        
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Missing_Count': missing[missing > 0],
                'Percentage': missing_pct[missing > 0]
            }).sort_values('Missing_Count', ascending=False)
            print(missing_df)
        else:
            print("No missing values found ✓")
        
        print("\n--- Summary Statistics (Numerical Features) ---")
        print(self.df[self.numerical_cols].describe().round(2))
        
        print("\n--- Duplicate Rows ---")
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
    
    
    def distribution_analysis(self):
        """
        Analyze and visualize distributions of numerical features
        """
        print("\n" + "="*80)
        print("📈 DISTRIBUTION ANALYSIS")
        print("="*80)
        
        if len(self.numerical_cols) == 0:
            print("No numerical columns to analyze")
            return
        
        # Calculate skewness
        skewness = self.df[self.numerical_cols].skew().sort_values(ascending=False)
        print("\n--- Skewness (|skew| > 1 indicates high skewness) ---")
        print(skewness.round(3))
        
        # Plot distributions
        n_cols = min(3, len(self.numerical_cols))
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(self.numerical_cols):
            if idx < len(axes):
                # Histogram with KDE
                axes[idx].hist(self.df[col].dropna(), bins=30, alpha=0.7, 
                              color='skyblue', edgecolor='black', density=True)
                
                # Add KDE line
                self.df[col].dropna().plot(kind='kde', ax=axes[idx], 
                                           color='red', linewidth=2)
                
                axes[idx].set_title(f'{col}\n(Skew: {skewness[col]:.2f})', 
                                   fontsize=10, fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Density')
                axes[idx].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(self.numerical_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
        
        # Q-Q plots for normality check
        print("\n--- Normality Check (Q-Q Plots) ---")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(self.numerical_cols):
            if idx < len(axes):
                stats.probplot(self.df[col].dropna(), dist="norm", plot=axes[idx])
                axes[idx].set_title(f'{col} - Q-Q Plot', fontsize=10, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        for idx in range(len(self.numerical_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    
    def correlation_analysis(self):
        """
        Analyze correlations between numerical features
        """
        print("\n" + "="*80)
        print("🔗 CORRELATION ANALYSIS")
        print("="*80)
        
        if len(self.numerical_cols) < 2:
            print("Need at least 2 numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numerical_cols].corr()
        
        # Find highly correlated pairs
        print("\n--- Highly Correlated Feature Pairs (|correlation| > 0.7) ---")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', 
                                                               key=abs, 
                                                               ascending=False)
            print(high_corr_df.to_string(index=False))
            print("\n⚠️  High correlation may indicate redundant features for clustering")
        else:
            print("No highly correlated pairs found (all |correlation| < 0.7)")
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap - Numerical Features', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    
    def outlier_detection(self):
        """
        Detect and visualize outliers using box plots and IQR method
        """
        print("\n" + "="*80)
        print("🎯 OUTLIER DETECTION")
        print("="*80)
        
        if len(self.numerical_cols) == 0:
            print("No numerical columns to analyze")
            return
        
        # Calculate outliers using IQR method
        outlier_summary = []
        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | 
                              (self.df[col] > upper_bound)][col]
            
            outlier_summary.append({
                'Feature': col,
                'Outlier_Count': len(outliers),
                'Outlier_Percentage': (len(outliers) / len(self.df)) * 100,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })
        
        outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Count', 
                                                               ascending=False)
        print("\n--- Outlier Summary (IQR Method) ---")
        print(outlier_df.to_string(index=False))
        
        # Box plots
        n_cols = min(3, len(self.numerical_cols))
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(self.numerical_cols):
            if idx < len(axes):
                bp = axes[idx].boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_edgecolor('black')
                axes[idx].set_title(f'{col}\n({outlier_df[outlier_df["Feature"]==col]["Outlier_Count"].values[0]} outliers)', 
                                   fontsize=10, fontweight='bold')
                axes[idx].set_ylabel(col)
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        for idx in range(len(self.numerical_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
        
        print("\n⚠️  Outliers can significantly impact unsupervised learning algorithms")
    
    
    def categorical_analysis(self):
        """
        Analyze categorical features
        """
        print("\n" + "="*80)
        print("📂 CATEGORICAL FEATURE ANALYSIS")
        print("="*80)
        
        if len(self.categorical_cols) == 0:
            print("No categorical columns to analyze")
            return
        
        # Cardinality analysis
        print("\n--- Cardinality (Unique Values) ---")
        cardinality = []
        for col in self.categorical_cols:
            n_unique = self.df[col].nunique()
            cardinality.append({
                'Feature': col,
                'Unique_Values': n_unique,
                'Percentage': (n_unique / len(self.df)) * 100
            })
        
        card_df = pd.DataFrame(cardinality).sort_values('Unique_Values', ascending=False)
        print(card_df.to_string(index=False))
        
        # Plot top categories for each categorical feature
        n_cols = min(2, len(self.categorical_cols))
        n_rows = (len(self.categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(self.categorical_cols):
            if idx < len(axes):
                # Get top 10 categories
                top_categories = self.df[col].value_counts().head(10)
                
                top_categories.plot(kind='barh', ax=axes[idx], color='coral', 
                                   edgecolor='black')
                axes[idx].set_title(f'{col} - Top Categories', 
                                   fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('Count')
                axes[idx].grid(True, alpha=0.3, axis='x')
        
        for idx in range(len(self.categorical_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    
    def variance_analysis(self):
        """
        Analyze feature variance to identify low-variance features
        """
        print("\n" + "="*80)
        print("📉 VARIANCE ANALYSIS")
        print("="*80)
        
        if len(self.numerical_cols) == 0:
            print("No numerical columns to analyze")
            return
        
        # Calculate variance
        variance = self.df[self.numerical_cols].var().sort_values(ascending=True)
        
        # Standardize features and calculate variance
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numerical_cols].fillna(0))
        scaled_df = pd.DataFrame(scaled_data, columns=self.numerical_cols)
        normalized_variance = scaled_df.var().sort_values(ascending=True)
        
        print("\n--- Normalized Variance (After StandardScaling) ---")
        print(normalized_variance.round(4))
        
        # Identify low variance features
        low_var_threshold = 0.01
        low_var_features = normalized_variance[normalized_variance < low_var_threshold]
        
        if len(low_var_features) > 0:
            print(f"\nFeatures with very low variance (< {low_var_threshold}):")
            print(low_var_features.index.tolist())
            print("Consider removing these features for unsupervised learning")
        else:
            print(f"\n✓ No features with extremely low variance (< {low_var_threshold})")
        
        # Variance bar plot
        plt.figure(figsize=(12, 6))
        normalized_variance.plot(kind='barh', color='teal', edgecolor='black')
        plt.xlabel('Variance (Normalized)')
        plt.ylabel('Features')
        plt.title('Feature Variance (After Standardization)', 
                 fontsize=14, fontweight='bold')
        plt.axvline(x=low_var_threshold, color='red', linestyle='--', 
                   label=f'Low Variance Threshold ({low_var_threshold})')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    
    def feature_relationships(self):
        """
        Visualize relationships between key numerical features
        """
        print("\n" + "="*80)
        print("🔍 FEATURE RELATIONSHIPS")
        print("="*80)
        
        if len(self.numerical_cols) < 2:
            print("Need at least 2 numerical columns for relationship analysis")
            return
        
        # Select top features for pairplot (limit to 5 for readability)
        n_features = min(5, len(self.numerical_cols))
        
        # Select features with highest variance
        top_features = self.df[self.numerical_cols].var().nlargest(n_features).index.tolist()
        
        print(f"\n--- Pairplot for Top {n_features} Features (by variance) ---")
        print(f"Features: {top_features}")
        
        # Create pairplot
        pairplot_df = self.df[top_features].copy()
        
        # Limit sample size if dataset is too large
        if len(pairplot_df) > 1000:
            print(f"(Sampling 1000 points from {len(pairplot_df)} for visualization)")
            pairplot_df = pairplot_df.sample(1000, random_state=42)
        
        sns.pairplot(pairplot_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Feature Relationships - Pairplot', 
                    y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    
    def pca_analysis(self):
        """
        Perform PCA to understand data dimensionality and variance
        """
        print("\n" + "="*80)
        print("🎨 PCA ANALYSIS")
        print("="*80)
        
        if len(self.numerical_cols) < 2:
            print("Need at least 2 numerical columns for PCA")
            return
        
        # Prepare data
        X = self.df[self.numerical_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        n_components = min(10, len(self.numerical_cols))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print("\n--- Explained Variance by Principal Components ---")
        for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
            print(f"PC{i+1}: {var:.4f} (Cumulative: {cum_var:.4f})")
        
        # Find number of components for 90% variance
        n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
        print(f"\n💡 {n_components_90} components explain 90% of variance")
        
        # Scree plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        axes[0].bar(range(1, n_components+1), explained_var, 
                   color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Scree Plot - Individual Variance', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Cumulative variance
        axes[1].plot(range(1, n_components+1), cumulative_var, 
                    marker='o', color='crimson', linewidth=2, markersize=8)
        axes[1].axhline(y=0.90, color='green', linestyle='--', 
                       label='90% Variance')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance', 
                         fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2D PCA visualization
        if X_pca.shape[1] >= 2:
            plt.figure(figsize=(10, 7))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, 
                       c=range(len(X_pca)), cmap='viridis', edgecolor='k', s=50)
            plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance)')
            plt.title('2D PCA Projection', fontsize=14, fontweight='bold')
            plt.colorbar(label='Sample Index')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Feature contributions to PC1 and PC2
        if pca.components_.shape[0] >= 2:
            plt.figure(figsize=(12, 5))
            
            components_df = pd.DataFrame(
                pca.components_[:2].T,
                columns=['PC1', 'PC2'],
                index=self.numerical_cols
            )
            
            components_df.plot(kind='barh', figsize=(12, max(6, len(self.numerical_cols)*0.3)))
            plt.xlabel('Component Loading')
            plt.ylabel('Features')
            plt.title('Feature Contributions to PC1 and PC2', 
                     fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linewidth=0.8)
            plt.legend(title='Component')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.show()



class UnsupervisedMLPipeline:
    """
    Automatic Unsupervised Machine Learning Pipeline for Automobile Datasets
    Applies clustering algorithms, dimensionality reduction, and provides comprehensive metrics
    """
    
    def __init__(self, df, n_clusters_range=(2, 10)):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed automobile dataset
        n_clusters_range : tuple
            Range of clusters to test (min, max)
        """
        self.df = df.copy()
        self.n_clusters_range = n_clusters_range
        self.numeric_df = None
        self.scaled_data = None
        self.optimal_k = None
        self.results = {}
        self.pca_components = None
        self.tsne_components = None
        
    def prepare_data(self):
        """Extract numeric features and scale data"""
        print("=" * 80)
        print("PREPARING DATA FOR UNSUPERVISED LEARNING")
        print("=" * 80)
        
        # Select only numeric columns
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        print(f"\n✓ Selected {self.numeric_df.shape[1]} numeric features")
        print(f"  Features: {list(self.numeric_df.columns)}")
        
        # Remove any remaining NaN values
        self.numeric_df = self.numeric_df.dropna()
        print(f"✓ Dataset shape after removing NaN: {self.numeric_df.shape}")
        
        # Scale the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.numeric_df)
        print("✓ Data scaled using StandardScaler")
        
        return self.scaled_data
    
    def find_optimal_clusters(self):
        """Find optimal number of clusters using Elbow Method and Silhouette Score"""
        print("\n" + "=" * 80)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 80)
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        k_range = range(self.n_clusters_range[0], self.n_clusters_range[1] + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, labels))
            calinski_scores.append(calinski_harabasz_score(self.scaled_data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.scaled_data, labels))
        
        # Find optimal k based on silhouette score
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"\n✓ Optimal number of clusters (based on Silhouette Score): {self.optimal_k}")
        print(f"  Silhouette Score: {max(silhouette_scores):.4f}")
        
        # Plot evaluation metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cluster Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # Elbow Method
        axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=11)
        axes[0, 0].set_title('Elbow Method', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        axes[0, 0].legend()
        
        # Silhouette Score
        axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[0, 1].set_ylabel('Silhouette Score', fontsize=11)
        axes[0, 1].set_title('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        axes[0, 1].legend()
        
        # Calinski-Harabasz Score
        axes[1, 0].plot(k_range, calinski_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[1, 0].set_ylabel('Calinski-Harabasz Score', fontsize=11)
        axes[1, 0].set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        axes[1, 0].legend()
        
        # Davies-Bouldin Score
        axes[1, 1].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[1, 1].set_ylabel('Davies-Bouldin Score', fontsize=11)
        axes[1, 1].set_title('Davies-Bouldin Score (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(x=self.optimal_k, color='r', linestyle='--', label=f'Optimal k={self.optimal_k}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return self.optimal_k
    
    def apply_kmeans(self):
        """Apply K-Means clustering"""
        print("\n" + "=" * 80)
        print("APPLYING K-MEANS CLUSTERING")
        print("=" * 80)
        
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.scaled_data)
        
        # Calculate metrics
        silhouette = silhouette_score(self.scaled_data, labels)
        calinski = calinski_harabasz_score(self.scaled_data, labels)
        davies_bouldin = davies_bouldin_score(self.scaled_data, labels)
        
        self.results['kmeans'] = {
            'model': kmeans,
            'labels': labels,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': self.optimal_k
        }
        
        print(f"\n✓ K-Means Clustering completed with k={self.optimal_k}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"\n  Cluster Distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"    Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def apply_dbscan(self):
        """Apply DBSCAN clustering"""
        print("\n" + "=" * 80)
        print("APPLYING DBSCAN CLUSTERING")
        print("=" * 80)
        
        # Find optimal eps using k-distance graph
        distances = cdist(self.scaled_data, self.scaled_data, metric='euclidean')
        k = 4  # typically use k = dimensions
        k_distances = np.sort(distances, axis=1)[:, k]
        k_distances = np.sort(k_distances)
        
        # Use knee point as eps (simplified approach)
        eps = np.percentile(k_distances, 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(self.scaled_data)
        
        # Calculate metrics (excluding noise points)
        if len(np.unique(labels[labels != -1])) > 1:
            mask = labels != -1
            silhouette = silhouette_score(self.scaled_data[mask], labels[mask])
            calinski = calinski_harabasz_score(self.scaled_data[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(self.scaled_data[mask], labels[mask])
        else:
            silhouette = calinski = davies_bouldin = -1
        
        n_clusters = len(np.unique(labels[labels != -1]))
        n_noise = np.sum(labels == -1)
        
        self.results['dbscan'] = {
            'model': dbscan,
            'labels': labels,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps
        }
        
        print(f"\n✓ DBSCAN Clustering completed")
        print(f"  Number of clusters found: {n_clusters}")
        print(f"  Number of noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        print(f"  Epsilon (eps): {eps:.4f}")
        if silhouette != -1:
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Calinski-Harabasz Score: {calinski:.4f}")
            print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        if n_clusters > 0:
            print(f"\n  Cluster Distribution:")
            unique, counts = np.unique(labels[labels != -1], return_counts=True)
            for cluster, count in zip(unique, counts):
                print(f"    Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def apply_hierarchical(self):
        """Apply Hierarchical (Agglomerative) clustering"""
        print("\n" + "=" * 80)
        print("APPLYING HIERARCHICAL CLUSTERING")
        print("=" * 80)
        
        hierarchical = AgglomerativeClustering(n_clusters=self.optimal_k)
        labels = hierarchical.fit_predict(self.scaled_data)
        
        # Calculate metrics
        silhouette = silhouette_score(self.scaled_data, labels)
        calinski = calinski_harabasz_score(self.scaled_data, labels)
        davies_bouldin = davies_bouldin_score(self.scaled_data, labels)
        
        self.results['hierarchical'] = {
            'model': hierarchical,
            'labels': labels,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': self.optimal_k
        }
        
        print(f"\n✓ Hierarchical Clustering completed with k={self.optimal_k}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"\n  Cluster Distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"    Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def apply_pca(self, n_components=2):
        """Apply PCA for dimensionality reduction"""
        print("\n" + "=" * 80)
        print("APPLYING PCA (DIMENSIONALITY REDUCTION)")
        print("=" * 80)
        
        pca = PCA(n_components=n_components)
        self.pca_components = pca.fit_transform(self.scaled_data)
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"\n✓ PCA completed with {n_components} components")
        print(f"  Explained variance by component:")
        for i, var in enumerate(explained_var):
            print(f"    PC{i+1}: {var*100:.2f}%")
        print(f"  Cumulative variance explained: {cumulative_var[-1]*100:.2f}%")
        
        self.results['pca'] = {
            'model': pca,
            'components': self.pca_components,
            'explained_variance_ratio': explained_var,
            'cumulative_variance': cumulative_var
        }
        
        return self.pca_components
    
    def apply_tsne(self, n_components=2):
        """Apply t-SNE for dimensionality reduction"""
        print("\n" + "=" * 80)
        print("APPLYING t-SNE (DIMENSIONALITY REDUCTION)")
        print("=" * 80)
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
        self.tsne_components = tsne.fit_transform(self.scaled_data)
        
        print(f"\n✓ t-SNE completed with {n_components} components")
        
        self.results['tsne'] = {
            'model': tsne,
            'components': self.tsne_components
        }
        
        return self.tsne_components
    
    def visualize_clusters(self):
        """Visualize clustering results"""
        print("\n" + "=" * 80)
        print("GENERATING CLUSTER VISUALIZATIONS")
        print("=" * 80)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. K-Means with PCA
        ax1 = plt.subplot(2, 3, 1)
        scatter1 = ax1.scatter(self.pca_components[:, 0], self.pca_components[:, 1], 
                               c=self.results['kmeans']['labels'], cmap='viridis', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('First Principal Component', fontsize=11)
        ax1.set_ylabel('Second Principal Component', fontsize=11)
        ax1.set_title(f'K-Means Clustering (PCA)\nSilhouette: {self.results["kmeans"]["silhouette_score"]:.3f}', 
                      fontsize=12, fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        ax1.grid(True, alpha=0.3)
        
        # 2. K-Means with t-SNE
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(self.tsne_components[:, 0], self.tsne_components[:, 1], 
                               c=self.results['kmeans']['labels'], cmap='viridis', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('t-SNE Component 1', fontsize=11)
        ax2.set_ylabel('t-SNE Component 2', fontsize=11)
        ax2.set_title(f'K-Means Clustering (t-SNE)\nSilhouette: {self.results["kmeans"]["silhouette_score"]:.3f}', 
                      fontsize=12, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        ax2.grid(True, alpha=0.3)
        
        # 3. DBSCAN with PCA
        ax3 = plt.subplot(2, 3, 3)
        scatter3 = ax3.scatter(self.pca_components[:, 0], self.pca_components[:, 1], 
                               c=self.results['dbscan']['labels'], cmap='viridis', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('First Principal Component', fontsize=11)
        ax3.set_ylabel('Second Principal Component', fontsize=11)
        sil_score = self.results['dbscan']['silhouette_score']
        sil_text = f"{sil_score:.3f}" if sil_score != -1 else "N/A"
        ax3.set_title(f'DBSCAN Clustering (PCA)\nSilhouette: {sil_text}', 
                      fontsize=12, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, label='Cluster')
        ax3.grid(True, alpha=0.3)
        
        # 4. Hierarchical with PCA
        ax4 = plt.subplot(2, 3, 4)
        scatter4 = ax4.scatter(self.pca_components[:, 0], self.pca_components[:, 1], 
                               c=self.results['hierarchical']['labels'], cmap='viridis', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('First Principal Component', fontsize=11)
        ax4.set_ylabel('Second Principal Component', fontsize=11)
        ax4.set_title(f'Hierarchical Clustering (PCA)\nSilhouette: {self.results["hierarchical"]["silhouette_score"]:.3f}', 
                      fontsize=12, fontweight='bold')
        plt.colorbar(scatter4, ax=ax4, label='Cluster')
        ax4.grid(True, alpha=0.3)
        
        # 5. Dendrogram
        ax5 = plt.subplot(2, 3, 5)
        # Sample data if too large
        sample_size = min(100, len(self.scaled_data))
        sample_indices = np.random.choice(len(self.scaled_data), sample_size, replace=False)
        linkage_matrix = linkage(self.scaled_data[sample_indices], method='ward')
        dendrogram(linkage_matrix, ax=ax5, leaf_font_size=8)
        ax5.set_xlabel('Sample Index', fontsize=11)
        ax5.set_ylabel('Distance', fontsize=11)
        ax5.set_title(f'Hierarchical Clustering Dendrogram\n(Sample of {sample_size} points)', 
                      fontsize=12, fontweight='bold')
        
        # 6. Cluster size comparison
        ax6 = plt.subplot(2, 3, 6)
        algorithms = ['K-Means', 'DBSCAN', 'Hierarchical']
        cluster_counts = [
            len(np.unique(self.results['kmeans']['labels'])),
            len(np.unique(self.results['dbscan']['labels'][self.results['dbscan']['labels'] != -1])),
            len(np.unique(self.results['hierarchical']['labels']))
        ]
        bars = ax6.bar(algorithms, cluster_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                       alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Number of Clusters', fontsize=11)
        ax6.set_title('Number of Clusters by Algorithm', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Cluster visualizations generated")
    
    def generate_cluster_profiles(self):
        """Generate profiles for each cluster"""
        print("\n" + "=" * 80)
        print("GENERATING CLUSTER PROFILES (K-MEANS)")
        print("=" * 80)
        
        # Add cluster labels to original dataframe
        df_with_clusters = self.numeric_df.copy()
        df_with_clusters['Cluster'] = self.results['kmeans']['labels']
        
        # Calculate cluster statistics
        cluster_profiles = df_with_clusters.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
        
        print("\n" + "=" * 80)
        print("CLUSTER PROFILES - Statistical Summary")
        print("=" * 80)
        print(cluster_profiles)
        
        # Visualize cluster profiles
        n_features = len(self.numeric_df.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, column in enumerate(self.numeric_df.columns):
            cluster_means = df_with_clusters.groupby('Cluster')[column].mean()
            axes[idx].bar(cluster_means.index, cluster_means.values, 
                         color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel('Cluster', fontsize=10)
            axes[idx].set_ylabel(f'Mean {column}', fontsize=10)
            axes[idx].set_title(f'{column} by Cluster', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return cluster_profiles
    
    def print_summary(self):
        """Print comprehensive summary of results"""
        print("\n" + "=" * 80)
        print("UNSUPERVISED LEARNING PIPELINE - SUMMARY")
        print("=" * 80)
        
        print("\n📊 DATASET INFORMATION:")
        print(f"  Total samples: {self.numeric_df.shape[0]}")
        print(f"  Total features: {self.numeric_df.shape[1]}")
        
        print("\n🔍 CLUSTERING RESULTS:")
        print("\n  Algorithm Performance Comparison:")
        print("  " + "-" * 76)
        print(f"  {'Algorithm':<20} {'Clusters':<12} {'Silhouette':<15} {'Calinski-H':<15} {'Davies-B':<15}")
        print("  " + "-" * 76)
        
        for algo_name in ['kmeans', 'dbscan', 'hierarchical']:
            result = self.results[algo_name]
            sil = f"{result['silhouette_score']:.4f}" if result['silhouette_score'] != -1 else "N/A"
            cal = f"{result['calinski_harabasz_score']:.2f}" if result['calinski_harabasz_score'] != -1 else "N/A"
            dav = f"{result['davies_bouldin_score']:.4f}" if result['davies_bouldin_score'] != -1 else "N/A"
            
            print(f"  {algo_name.upper():<20} {result['n_clusters']:<12} {sil:<15} {cal:<15} {dav:<15}")
        
        print("  " + "-" * 76)
        
        print("\n📈 DIMENSIONALITY REDUCTION:")
        print(f"  PCA - Variance explained: {self.results['pca']['cumulative_variance'][-1]*100:.2f}%")
        print(f"  t-SNE - Components: 2D projection completed")
        
        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    
    def run_pipeline(self):
        """Run the complete unsupervised ML pipeline"""
        print("\n" + "🚀" * 40)
        print(" " * 20 + "UNSUPERVISED ML PIPELINE STARTED")
        print("🚀" * 40 + "\n")
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Find optimal clusters
        self.find_optimal_clusters()
        
        # Step 3: Apply clustering algorithms
        self.apply_kmeans()
        self.apply_dbscan()
        self.apply_hierarchical()
        
        # Step 4: Apply dimensionality reduction
        self.apply_pca()
        self.apply_tsne()
        
        # Step 5: Visualize results
        self.visualize_clusters()
        
        # Step 6: Generate cluster profiles
        self.generate_cluster_profiles()
        
        # Step 7: Print summary
        self.print_summary()
        
        return self.results
