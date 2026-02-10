import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings
warnings.filterwarnings('ignore')


def plot_time_series(df, target_col, figsize=(15, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df.index, df[target_col], linewidth=1.5, color='#2E86AB')
    ax.set_title(f'Time Series Plot: {target_col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

#'''
def plot_seasonal_decomposition(df, target_col, period=None, figsize=(15, 10), save_path=None):
    if period is None:
        freq = pd.infer_freq(df.index)
        if freq and 'D' in freq:
            period = 7
        elif freq and 'W' in freq:
            period = 4
        elif freq and 'M' in freq:
            period = 12
        elif freq and 'H' in freq:
            period = 24
        else:
            period = 7
    
    if len(df) < 2 * period:
        print(f"Not enough data for decomposition. Need at least {2 * period} points.")
        return None
    
    try:
        decomposition = seasonal_decompose(df[target_col], model='additive', period=period, extrapolate_trend='freq')
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        decomposition.observed.plot(ax=axes[0], color='#2E86AB')
        axes[0].set_ylabel('Observed', fontsize=11)
        axes[0].set_title('Seasonal Decomposition', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        decomposition.trend.plot(ax=axes[1], color='#A23B72')
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        decomposition.seasonal.plot(ax=axes[2], color='#F18F01')
        axes[2].set_ylabel('Seasonal', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        decomposition.resid.plot(ax=axes[3], color='#C73E1D')
        axes[3].set_ylabel('Residual', fontsize=11)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return decomposition
    
    except Exception as e:
        print(f"Decomposition failed: {str(e)}")
        return None
#'''

def plot_autocorrelation(df, target_col, lags=25, figsize=(15, 5), save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_acf(df[target_col].dropna(), lags=lags, ax=axes[0], color='#2E86AB')
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    plot_pacf(df[target_col].dropna(), lags=lags, ax=axes[1], color='#A23B72')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_distribution(df, target_col, figsize=(15, 5), save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].hist(df[target_col].dropna(), bins=50, edgecolor='black', color='#2E86AB', alpha=0.7)
    axes[0].set_title('Histogram', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(target_col, fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    df[target_col].dropna().plot(kind='box', ax=axes[1], color='#2E86AB')
    axes[1].set_title('Box Plot', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(target_col, fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_rolling_statistics(df, target_col, windows=[7, 30], figsize=(15, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df.index, df[target_col], label='Original', linewidth=1, alpha=0.6, color='gray')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for i, window in enumerate(windows):
        rolling_mean = df[target_col].rolling(window=window).mean()
        rolling_std = df[target_col].rolling(window=window).std()
        
        ax.plot(df.index, rolling_mean, label=f'Rolling Mean ({window})', 
                linewidth=2, color=colors[i % len(colors)])
        ax.fill_between(df.index, 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.2, color=colors[i % len(colors)])
    
    ax.set_title('Rolling Statistics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_feature_correlation(df, target_col, figsize=(12, 10), save_path=None):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    if target_col in numeric_cols:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, max(6, len(target_corr) * 0.3)))
        
        colors = ['#2E86AB' if x > 0 else '#C73E1D' for x in target_corr.values]
        ax2.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(target_corr)))
        ax2.set_yticklabels(target_corr.index)
        ax2.set_xlabel('Correlation Coefficient', fontsize=12)
        ax2.set_title(f'Features Correlation with {target_col}', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            save_path_target = save_path.replace('.png', '_target.png')
            plt.savefig(save_path_target, dpi=150, bbox_inches='tight')
        plt.show()
        
        return corr_matrix, target_corr
    
    return corr_matrix


def plot_lag_features(df, target_col, lags=[1, 7, 30], figsize=(15, 10), save_path=None):
    n_lags = len(lags)
    n_cols = 3
    n_rows = (n_lags + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_lags > 1 else [axes]
    
    for i, lag in enumerate(lags):
        if i < len(axes):
            lagged_values = df[target_col].shift(lag)
            
            axes[i].scatter(lagged_values, df[target_col], alpha=0.5, s=20, color='#2E86AB')
            axes[i].set_xlabel(f'{target_col} (t-{lag})', fontsize=11)
            axes[i].set_ylabel(f'{target_col} (t)', fontsize=11)
            axes[i].set_title(f'Lag {lag} Scatter Plot', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            valid_mask = ~(lagged_values.isna() | df[target_col].isna())
            if valid_mask.sum() > 0:
                correlation = np.corrcoef(lagged_values[valid_mask], df[target_col][valid_mask])[0, 1]
                axes[i].text(0.05, 0.95, f'Corr: {correlation:.3f}', 
                           transform=axes[i].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_time_based_patterns(df, target_col, figsize=(15, 12), save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    if 'month' in df.columns:
        monthly_avg = df.groupby('month')[target_col].mean()
        axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='#2E86AB', alpha=0.7)
        axes[0, 0].set_title('Average by Month', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Month', fontsize=11)
        axes[0, 0].set_ylabel(f'Average {target_col}', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[0, 0].text(0.5, 0.5, 'No month data available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    if 'dayofweek' in df.columns:
        dow_avg = df.groupby('dayofweek')[target_col].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(len(dow_avg)), dow_avg.values, color='#A23B72', alpha=0.7)
        axes[0, 1].set_xticks(range(len(dow_avg)))
        axes[0, 1].set_xticklabels([days[i] if i < len(days) else str(i) for i in dow_avg.index])
        axes[0, 1].set_title('Average by Day of Week', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week', fontsize=11)
        axes[0, 1].set_ylabel(f'Average {target_col}', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[0, 1].text(0.5, 0.5, 'No day of week data available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    if 'quarter' in df.columns:
        quarter_avg = df.groupby('quarter')[target_col].mean()
        axes[1, 0].bar(quarter_avg.index, quarter_avg.values, color='#F18F01', alpha=0.7)
        axes[1, 0].set_title('Average by Quarter', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Quarter', fontsize=11)
        axes[1, 0].set_ylabel(f'Average {target_col}', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'No quarter data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    if 'is_weekend' in df.columns:
        weekend_avg = df.groupby('is_weekend')[target_col].mean()
        labels = ['Weekday', 'Weekend']
        axes[1, 1].bar(range(len(weekend_avg)), weekend_avg.values, color='#C73E1D', alpha=0.7)
        axes[1, 1].set_xticks(range(len(weekend_avg)))
        axes[1, 1].set_xticklabels([labels[i] if i < len(labels) else str(i) for i in weekend_avg.index])
        axes[1, 1].set_title('Average: Weekday vs Weekend', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel(f'Average {target_col}', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'No weekend data available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    if 'year' in df.columns:
        yearly_avg = df.groupby('year')[target_col].mean()
        axes[2, 0].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, 
                       markersize=8, color='#2E86AB')
        axes[2, 0].set_title('Average by Year', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Year', fontsize=11)
        axes[2, 0].set_ylabel(f'Average {target_col}', fontsize=11)
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'No year data available', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
    
    if hasattr(df.index, 'hour'):
        hourly_avg = df.groupby(df.index.hour)[target_col].mean()
        axes[2, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, 
                       markersize=6, color='#A23B72')
        axes[2, 1].set_title('Average by Hour', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Hour', fontsize=11)
        axes[2, 1].set_ylabel(f'Average {target_col}', fontsize=11)
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'No hourly data available', 
                       ha='center', va='center', transform=axes[2, 1].transAxes)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def calculate_statistical_summary(df, target_col):
    series = df[target_col].dropna()
    
    summary = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'cv': (series.std() / series.mean()) * 100 if series.mean() != 0 else np.nan
    }
    
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ['Value']
    
    print("="*60)
    print(f"STATISTICAL SUMMARY: {target_col}")
    print("="*60)
    print(summary_df.to_string())
    print("="*60)
    
    return summary_df


def detect_outliers(df, target_col, method='iqr', threshold=1.5):
    series = df[target_col].dropna()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series))
        outliers = z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    n_outliers = outliers.sum()
    pct_outliers = (n_outliers / len(series)) * 100
    
    print("\n" + "="*60)
    print(f"OUTLIER DETECTION: {target_col}")
    print("="*60)
    print(f"Method: {method.upper()}")
    print(f"Threshold: {threshold}")
    print(f"Total outliers: {n_outliers}")
    print(f"Percentage: {pct_outliers:.2f}%")
    
    if method == 'iqr':
        print(f"Lower bound: {lower_bound:.4f}")
        print(f"Upper bound: {upper_bound:.4f}")
    
    print("="*60)
    
    return outliers, n_outliers


def analyze_stationarity(df, target_col):
    from statsmodels.tsa.stattools import adfuller
    
    series = df[target_col].dropna()
    
    result = adfuller(series)
    
    print("\n" + "="*60)
    print(f"AUGMENTED DICKEY-FULLER TEST: {target_col}")
    print("="*60)
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.6f}")
    
    if result[1] <= 0.05:
        print("\nConclusion: Series is stationary (p-value <= 0.05)")
    else:
        print("\nConclusion: Series is non-stationary (p-value > 0.05)")
        print("Consider differencing or other transformations")
    
    print("="*60)
    
    return result


def plot_feature_importance_correlation(df, target_col, top_n=15, figsize=(10, 8), save_path=None):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if target_col not in numeric_cols:
        print(f"{target_col} is not numeric")
        return None
    
    correlations = df[numeric_cols].corrwith(df[target_col]).drop(target_col)
    correlations = correlations.abs().sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2E86AB' if x > 0 else '#C73E1D' for x in correlations.values]
    ax.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index)
    ax.set_xlabel('Absolute Correlation', fontsize=12)
    ax.set_title(f'Top {top_n} Features by Correlation with {target_col}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return correlations


def plot_pairwise_relationships(df, target_col, top_features=5, figsize=(15, 12), save_path=None):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if target_col not in numeric_cols:
        print(f"{target_col} is not numeric")
        return None
    
    correlations = df[numeric_cols].corrwith(df[target_col]).drop(target_col)
    top_cols = correlations.abs().sort_values(ascending=False).head(top_features).index.tolist()
    
    n_features = len(top_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(top_cols):
        if i < len(axes):
            axes[i].scatter(df[feature], df[target_col], alpha=0.5, s=20, color='#2E86AB')
            axes[i].set_xlabel(feature, fontsize=11)
            axes[i].set_ylabel(target_col, fontsize=11)
            axes[i].set_title(f'{feature} vs {target_col}', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            corr = correlations[feature]
            axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                       transform=axes[i].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def comprehensive_eda(df, target_col, output_dir='./eda_output'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TIME SERIES EDA")
    print("="*80)
    
    print("\n1. Statistical Summary")
    stats_summary = calculate_statistical_summary(df, target_col)
    
    print("\n2. Time Series Plot")
    plot_time_series(df, target_col, save_path=f'{output_dir}/01_time_series.png')
    
    print("\n3. Distribution Analysis")
    plot_distribution(df, target_col, save_path=f'{output_dir}/02_distribution.png')
    
    print("\n4. Outlier Detection")
    outliers_iqr, n_outliers_iqr = detect_outliers(df, target_col, method='iqr')
    
    print("\n5. Stationarity Test")
    try:
        adf_result = analyze_stationarity(df, target_col)
    except Exception as e:
        print(f"Stationarity test failed: {str(e)}")
    
    print("\n6. Seasonal Decomposition")
    decomposition = plot_seasonal_decomposition(df, target_col, save_path=f'{output_dir}/03_decomposition.png')
    
    print("\n7. Autocorrelation Analysis")
    plot_autocorrelation(df, target_col, save_path=f'{output_dir}/04_autocorrelation.png')
    
    print("\n8. Rolling Statistics")
    plot_rolling_statistics(df, target_col, save_path=f'{output_dir}/05_rolling_stats.png')
    
    print("\n9. Lag Features Analysis")
    plot_lag_features(df, target_col, save_path=f'{output_dir}/06_lag_features.png')
    
    print("\n10. Time-based Patterns")
    plot_time_based_patterns(df, target_col, save_path=f'{output_dir}/07_time_patterns.png')
    
    print("\n11. Feature Correlation")
    corr_matrix = plot_feature_correlation(df, target_col, save_path=f'{output_dir}/08_correlation.png')
    
    print("\n12. Feature Importance by Correlation")
    plot_feature_importance_correlation(df, target_col, save_path=f'{output_dir}/09_feature_importance.png')
    
    print("\n13. Pairwise Relationships")
    plot_pairwise_relationships(df, target_col, save_path=f'{output_dir}/10_pairwise.png')
    
    print("\n" + "="*80)
    print("EDA COMPLETE")
    print(f"All plots saved to: {output_dir}/")
    print("="*80)
    
    return {
        'statistical_summary': stats_summary,
        'outliers': outliers_iqr,
        'n_outliers': n_outliers_iqr,
        'correlation_matrix': corr_matrix
    }

def eda_ts(df, target):
    result = comprehensive_eda(df, target_col=target, output_dir='./eda_output')
    return result