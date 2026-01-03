import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import sys

def main():
    # Load the data
    try:
        df = pd.read_csv('data_new.csv')
    except FileNotFoundError:
        print("Error: data_new.csv not found.")
        sys.exit(1)

    # Define features to use
    # Mapping user requested 'Loudness' to 'Loud (Db)' in the CSV
    feature_mapping = {
        'BPM': 'BPM',
        'Energy': 'Energy',
        'Dance': 'Dance',
        'Valence': 'Valence',
        'Acoustic': 'Acoustic',
        'Instrumental': 'Instrumental',
        'Speech': 'Speech',
        'Live': 'Live',
        'Loudness': 'Loud (Db)'
    }
    
    features = list(feature_mapping.values())
    
    # Check if columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns are missing in the CSV: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Subset the data for PCA, but keep the original df aligned for hover info
    # We drop rows with missing values in the feature columns from BOTH
    initial_shape = df.shape
    df_clean = df.dropna(subset=features).copy()
    if df_clean.shape[0] < initial_shape[0]:
        print(f"Warning: Dropped {initial_shape[0] - df_clean.shape[0]} rows due to missing values in selected features.")

    data_subset = df_clean[features]

    # Preprocess: Standardize the data
    # PCA is sensitive to scale, so we normalize the features.
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_subset)

    # Run PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data_scaled)

    # Explain components
    # Get components (loadings)
    components = pca.components_
    print("Principal Components (Loadings):\n", components)

    # Create a DataFrame for better interpretation
    pc_df = pd.DataFrame(components, columns=features, index=['PC1', 'PC2', 'PC3'])
    print("\nInterpretable Components:\n", pc_df)


    # Create a DataFrame with the PCA results and add back metadata
    df_clean['PC1'] = principal_components[:, 0]
    df_clean['PC2'] = principal_components[:, 1]
    df_clean['PC3'] = principal_components[:, 2]
    
    # Prepare hover data
    hover_data = []
    if 'Song' in df_clean.columns: hover_data.append('Song')
    if 'Artist' in df_clean.columns: hover_data.append('Artist')
    if 'Genres' in df_clean.columns: hover_data.append('Genres')
    if 'Contributor' in df_clean.columns: hover_data.append('Contributor')

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio: {explained_variance}")
    print(f"Total Explained Variance: {sum(explained_variance):.2f}")

    # Visualization using Plotly
    fig = px.scatter_3d(
        df_clean, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        hover_data=hover_data,
        color='Contributor',
        title='3D PCA of Audio Features',
        labels={
            'PC1': f'PC1 ({explained_variance[0]:.2%} var)',
            'PC2': f'PC2 ({explained_variance[1]:.2%} var)',
            'PC3': f'PC3 ({explained_variance[2]:.2%} var)'
        }
    )
    
    # Improve marker layout
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # Save the plot
    output_file = 'pca_3d_plot.html'
    fig.write_html(output_file)
    print(f"3D PCA interactive plot saved to {output_file}")
    
    # Save PCA data to CSV
    pca_output_csv = 'pca_results.csv'
    df_clean.to_csv(pca_output_csv, index=False)
    print(f"PCA results saved to {pca_output_csv}")

    # Save explained variance
    pd.DataFrame({'ratio': explained_variance}).to_csv('pca_variance.csv', index=False)
    print("Explained variance ratio saved to pca_variance.csv")

if __name__ == "__main__":
    main()
