import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np
import sys

# Initialize the app
app = dash.Dash(__name__)

app.title = "Chancla Mix Telescope 🔭"

# --- Data Loading (Pre-calculated) ---
try:
    df_clean = pd.read_csv('pca_results.csv')
    explained_variance = pd.read_csv('pca_variance.csv')['ratio'].tolist()
except FileNotFoundError:
    print("Error: Pre-calculated PCA data not found. Run 'pca_analysis.py' first.")
    # Fallback to empty to avoid crashing immediately, though UI will be broken
    df_clean = pd.DataFrame()
    explained_variance = [0, 0, 0]

if not df_clean.empty:
    # Fill NaN genres/contributors for UI safety
    if 'Genres' in df_clean.columns:
        df_clean['Genres'] = df_clean['Genres'].fillna('Unknown')
    if 'Contributor' in df_clean.columns:
        df_clean['Contributor'] = df_clean['Contributor'].fillna('Unknown')

    # Prepare options for Dropdowns
    if 'Contributor' in df_clean.columns:
        unique_contributors = sorted(df_clean['Contributor'].unique().astype(str))
        contributor_options = [{'label': c, 'value': c} for c in unique_contributors]
        
        # Consistent Color Mapping for Contributors
        color_palette = px.colors.qualitative.Alphabet
        # Remove 7, 8, and 18
        color_palette = color_palette[:7] + color_palette[9:18] + color_palette[20:]
        contributor_color_map = {
            cont: color_palette[i % len(color_palette)] 
            for i, cont in enumerate(unique_contributors)
        }
    else:
        contributor_options = []
        contributor_color_map = {}

    # Prepare Genre options - Genres can be comma separated list "Pop, Rock"
    # We want individual genres.
    unique_genres = set()
    if 'Genres' in df_clean.columns:
        for g_str in df_clean['Genres'].astype(str):
            parts = [p.strip() for p in g_str.split(',')]
            unique_genres.update(parts)

    # Prepare Song search options
    # We display "Song - Artist" for clarity
    if 'Song' in df_clean.columns and 'Artist' in df_clean.columns:
        song_options = [
            {'label': f"{row['Song']} - {row['Artist']}", 'value': row['Spotify Track Id']} 
            for _, row in df_clean.iterrows()
        ]
        # Sort by song name
        song_options = sorted(song_options, key=lambda x: x['label'].lower())
    else:
        song_options = []

    genre_options = [{'label': g, 'value': g} for g in sorted(unique_genres) if g]


# --- App Layout ---
app.layout = html.Div(className='app-container', children=[
    
    html.Div([
        # Theme State
        html.Div(id='theme-dummy', style={'display': 'none'}),
        
        # Controls Column (Sidebar)
        html.Div([
            html.Img(src=app.get_asset_url('ChanclaIcon.svg'), style={'width': '100px', 'height': '100px', 'display': 'block', 'margin': 'auto', 'marginTop': '10px'}),
            html.H1("2025 Chancla Mix", style={
                'textAlign': 'center', 
                'fontSize': '24px', 
                'background': 'linear-gradient(to right, #DD3C00, #FCE49C)',
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'backgroundClip': 'text',
                'color': 'transparent',
                'fontWeight': 'bold',
                'marginBottom':'-15px'
            }),
            html.H1("Telescope 🔭", style={'textAlign': 'center', 'fontSize': '24px', 'color': '#aaaaaa'}),
            
            html.Label("Search for Song 🔎", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='song-search',
                options=song_options,
                value=None,
                placeholder="Search song title or artist...",
                searchable=True,
                clearable=True,
                optionHeight=50
            ),

            html.Br(),

            html.Label("Filter by Contributor 🎤", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='contributor-filter',
                options=contributor_options,
                value=[],
                multi=True,
                placeholder="Select Contributors..."
            ),
            
            html.Br(),
            
            html.Label("Filter by Genre 💿", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='genre-filter',
                options=genre_options,
                value=[],
                multi=True,
                placeholder="Select Genres (Match any)..."
            ),
            
            html.Br(),
            
            html.Label("Color By 🎨", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='color-selector',
                options=[
                    {'label': 'Contributor', 'value': 'Contributor'},
                    {'label': 'Popularity', 'value': 'Popularity'},
                    {'label': 'Energy', 'value': 'Energy'},
                    {'label': 'Dance', 'value': 'Dance'},
                    {'label': 'BPM', 'value': 'BPM'},
                    {'label': 'Valence', 'value': 'Valence'},
                    {'label': 'Acoustic', 'value': 'Acoustic'},
                    {'label': 'Instrumental', 'value': 'Instrumental'},
                    {'label': 'Speech', 'value': 'Speech'},
                    {'label': 'Live', 'value': 'Live'},
                    {'label': 'Loudness', 'value': 'Loudness'},
                    {'label': 'PC1 (Main Component)', 'value': 'PC1'},
                ],
                value='Contributor',
                clearable=False
            ),

            html.Br(),
            html.Label("Statistics 📊", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            html.Div(id='stats-output'),
            dcc.Checklist(
                    id='theme-toggle',
                    options=[{'label': ' Dark Mode 🌙', 'value': 'dark'}],
                    value=['dark'],
                    style={'fontSize': '12px', 'marginBottom': '10px'}
                ),
            html.Img(src=app.get_asset_url('ChanclaLogo.svg'), style={'width': '125px', 'height': '125px', 'display': 'block', 'margin': 'auto', 'marginTop': '20px'}),
            
        ], className='sidebar', style={
            'width': '300px', 
            'minWidth': '300px',
            'height': '100vh', 
            'padding': '20px', 
            'boxSizing': 'border-box', 
            'overflowY': 'auto'
        }),
        
        # Graph Column (Main Content)
        html.Div([
            dcc.Graph(id='pca-3d-graph', style={'height': '100vh', 'width': '100%'})
        ], style={'flexGrow': '1', 'height': '100vh'})
    ], id='main-container', style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'height': '100vh', 'overflow': 'hidden'})
])

# --- Clientside Callback for CSS Theme ---
app.clientside_callback(
    """
    function(themeValue) {
        const theme = (themeValue && themeValue.length > 0) ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', theme);
        return '';
    }
    """,
    Output('theme-dummy', 'children'),
    Input('theme-toggle', 'value')
)


# --- Callbacks ---
@app.callback(
    [Output('pca-3d-graph', 'figure'),
     Output('stats-output', 'children')],
    [Input('contributor-filter', 'value'),
     Input('genre-filter', 'value'),
     Input('color-selector', 'value'),
     Input('song-search', 'value'),
     Input('theme-toggle', 'value')]
)
def update_graph(selected_contributors, selected_genres, color_col, searched_song_id, theme_toggle):
    is_dark = theme_toggle and 'dark' in theme_toggle
    template = 'plotly_dark' if is_dark else 'plotly_white'
    
    if df_clean.empty:
        return {}, "No data loaded."
        
    # We always use the full clean dataset for plotting to enable "highlighting" instead of "isolating"
    plot_df = df_clean.copy()
    
    # Identify the "Filtered" subset (what would have been isolated before)
    is_filtered = pd.Series(True, index=plot_df.index)
    
    if selected_contributors:
        is_filtered &= plot_df['Contributor'].isin(selected_contributors)
        
    if selected_genres:
        pattern = '|'.join([g for g in selected_genres])
        is_filtered &= plot_df['Genres'].astype(str).str.contains(pattern, case=False, na=False)

    # Determine hover data
    hover_cols = []
    for col in ['Song', 'Artist', 'Genres', 'Contributor']:
        if col in plot_df.columns:
            hover_cols.append(col)

    # Determine visibility split
    matched_df = plot_df[is_filtered].copy()
    others_df = plot_df[~is_filtered].copy()
    
    filters_active = bool(selected_contributors or selected_genres)

    # Initialize plot with the complete dataset to ensure consistent coordinate space and color scales
    # We use matched_df as the primary trace if others_df is empty, otherwise we layer them.
    if not others_df.empty:
        # Create background trace first (others)
        fig = px.scatter_3d(
            others_df, 
            x='PC1', y='PC2', z='PC3',
            color=color_col if color_col in others_df.columns else 'PC1',
            color_discrete_map=contributor_color_map if color_col == 'Contributor' else None,
            hover_data=hover_cols,
            template=template,
            labels={
                'PC1': f'PC1 ({explained_variance[0]:.2%} var)',
                'PC2': f'PC2 ({explained_variance[1]:.2%} var)',
                'PC3': f'PC3 ({explained_variance[2]:.2%} var)'
            }
        )
        fig.update_traces(marker=dict(size=3, opacity=0.1), name='Other Tracks')
        
        # Add the matched/highlighted tracks as the foreground trace
        if not matched_df.empty:
            match_fig = px.scatter_3d(
                matched_df, 
                x='PC1', y='PC2', z='PC3',
                color=color_col if color_col in matched_df.columns else 'PC1',
                color_discrete_map=contributor_color_map if color_col == 'Contributor' else None,
                hover_data=hover_cols,
                template=template
            )
            # Add ALL traces from match_fig (one trace per category in color_col)
            for trace in match_fig.data:
                trace.marker.size = 6
                trace.marker.opacity = 0.8
                # Append to trace name to distinguish
                trace.name = f"{trace.name} (Matched)"
                fig.add_trace(trace)
    else:
        # If no tracks are in "others", just plot everything as matched
        fig = px.scatter_3d(
            matched_df, 
            x='PC1', y='PC2', z='PC3',
            color=color_col if color_col in matched_df.columns else 'PC1',
            color_discrete_map=contributor_color_map if color_col == 'Contributor' else None,
            hover_data=hover_cols,
            template=template,
            labels={
                'PC1': f'PC1 ({explained_variance[0]:.2%} var)',
                'PC2': f'PC2 ({explained_variance[1]:.2%} var)',
                'PC3': f'PC3 ({explained_variance[2]:.2%} var)'
            }
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8), name='All Tracks')

    # --- Centroids Logic ---
    # Determine which contributors to show centroids for
    if filters_active:
        # If filtering, only show centroids for relevant people
        relevant_contributors = selected_contributors if selected_contributors else matched_df['Contributor'].unique()
    else:
        # If no filters, show centroids for everyone
        relevant_contributors = plot_df['Contributor'].unique()

    # Identify if ONLY genre filter is active
    only_genre_filter = bool(selected_genres and not selected_contributors and not searched_song_id)

    stats_text = [html.P(f"Showing {len(matched_df)} tracks", style={'fontStyle':'italic'})]

    if only_genre_filter:
        counts = matched_df['Contributor'].value_counts()
        if not counts.empty:
            stats_text.append(html.H4("Songs by Contributor 🎤"))
            # Sort contributors by count (highest first)
            for cont, count in counts.items():
                stats_text.append(html.P(f"• {cont}: {count} tracks", style={'fontSize': '14px', 'marginLeft': '10px'}))
        
        # We still want to show centroids on the graph if we are filtering by genre
        # but the user requested "Do not show similarity view or the overall stats"
        # so we will calculate centroids for the graph but skip the distance text.
    
    if len(relevant_contributors) > 0 and 'Contributor' in plot_df.columns:
        centroids = []
        for cont in relevant_contributors:
            # We use plot_df for calculations to ensure we get the full average even if songs are faded
            cont_data = plot_df[plot_df['Contributor'] == cont]
            if cont_data.empty: continue
            
            c_x = cont_data['PC1'].mean()
            c_y = cont_data['PC2'].mean()
            c_z = cont_data['PC3'].mean()
            centroids.append({'Contributor': cont, 'PC1': c_x, 'PC2': c_y, 'PC3': c_z})
        
        if centroids:
            centroid_df = pd.DataFrame(centroids)
            # Add Centroid Trace
            centroid_fig = px.scatter_3d(
                centroid_df,
                x='PC1', y='PC2', z='PC3',
                color='Contributor',
                color_discrete_map=contributor_color_map,
                text='Contributor',
                template=template
            )
            centroid_trace = centroid_fig.data[0]
            
            # Apply common centroid styling while keeping mapped colors
            # If multiple contributors, Plotly creates multiple traces. We need to handle that.
            for trace in centroid_fig.data:
                trace.marker.symbol = 'diamond'
                trace.marker.size = 12
                trace.marker.line = dict(width=2, color='white')
                trace.name = f"{trace.name} (Centroid)"
                fig.add_trace(trace)

            # --- Similarity & Extremes Section ---
            # SKIP this section if only_genre_filter is active
            if not only_genre_filter:
                # Distances/Similarities logic
                if selected_contributors and len(selected_contributors) == 1:
                    # Single person selected: Show similarity against everyone else
                    target_cont = selected_contributors[0]
                    target_centroid = next(c for c in centroids if c['Contributor'] == target_cont)
                    
                    # We need all other centroids for comparison
                    other_contributors = [c for c in plot_df['Contributor'].unique() if c != target_cont]
                    all_other_centroids = []
                    for cont in other_contributors:
                        cont_data = plot_df[plot_df['Contributor'] == cont]
                        if cont_data.empty: continue
                        all_other_centroids.append({
                            'Contributor': cont,
                            'PC1': cont_data['PC1'].mean(),
                            'PC2': cont_data['PC2'].mean(),
                            'PC3': cont_data['PC3'].mean()
                        })
                    
                    if all_other_centroids:
                        stats_text.append(html.H4(f"Similarities for {target_cont}:"))
                        # Sort by distance (closest first)
                        comparisons = []
                        for c2 in all_other_centroids:
                            dist = np.sqrt((target_centroid['PC1']-c2['PC1'])**2 + (target_centroid['PC2']-c2['PC2'])**2 + (target_centroid['PC3']-c2['PC3'])**2)
                            similarity = 100 / (1 + dist)
                            comparisons.append({'cont': c2['Contributor'], 'dist': dist, 'sim': similarity})
                        
                        comparisons.sort(key=lambda x: x['dist'])
                        for comp in comparisons:
                            stats_text.append(html.P(f"{target_cont} 🔁 {comp['cont']}: {comp['dist']:.2f} ({comp['sim']:.1f}%)"))

                    # Contributor-specific extremes
                    cont_songs = plot_df[plot_df['Contributor'] == target_cont].copy()
                    if not cont_songs.empty:
                        # Distances to GLOBAL centroid
                        g_x = plot_df['PC1'].mean()
                        g_y = plot_df['PC2'].mean()
                        g_z = plot_df['PC3'].mean()
                        cont_songs['dist_to_global'] = np.sqrt((cont_songs['PC1']-g_x)**2 + (cont_songs['PC2']-g_y)**2 + (cont_songs['PC3']-g_z)**2)
                        
                        c_agreeable = cont_songs.loc[cont_songs['dist_to_global'].idxmin()]
                        c_polarizing = cont_songs.loc[cont_songs['dist_to_global'].idxmax()]
                        
                        stats_text.append(html.H4(f"{target_cont}'s Taste 🎤:"))
                        stats_text.append(html.P([
                            html.Span("Most Agreeable: ", style={'fontWeight': 'bold', 'color': '#28a745'}),
                            html.A(f"{c_agreeable['Song']} - {c_agreeable['Artist']}", 
                                   href=f"https://open.spotify.com/track/{c_agreeable['Spotify Track Id']}", 
                                   target="_blank", className='spotify-link')
                        ], style={'fontSize': '14px'}))
                        stats_text.append(html.P([
                            html.Span("Most Divergent: ", style={'fontWeight': 'bold', 'color': '#dc3545'}),
                            html.A(f"{c_polarizing['Song']} - {c_polarizing['Artist']}", 
                                   href=f"https://open.spotify.com/track/{c_polarizing['Spotify Track Id']}", 
                                   target="_blank", className='spotify-link')
                        ], style={'fontSize': '14px'}))

                elif len(centroids) > 1 and len(centroids) <= 5:
                    # Multiple people selected (2-5): Show internal similarities
                    stats_text.append(html.H4("Similarities:"))
                    for i in range(len(centroids)):
                        for j in range(i + 1, len(centroids)):
                            c1 = centroids[i]
                            c2 = centroids[j]
                            dist = np.sqrt((c1['PC1']-c2['PC1'])**2 + (c1['PC2']-c2['PC2'])**2 + (c1['PC3']-c2['PC3'])**2)
                            similarity = 100 / (1 + dist) 
                            stats_text.append(html.P(f"{c1['Contributor']} 🔁 {c2['Contributor']}: {dist:.2f} ({similarity:.1f}%)"))
                            
                            # Find songs from each person closest to the other person's centroid
                            c1_songs = plot_df[plot_df['Contributor'] == c1['Contributor']].copy()
                            c2_songs = plot_df[plot_df['Contributor'] == c2['Contributor']].copy()
                            
                            if not c1_songs.empty and not c2_songs.empty:
                                # Song from c1 closest to c2's centroid
                                c1_songs['dist_to_c2'] = np.sqrt((c1_songs['PC1']-c2['PC1'])**2 + (c1_songs['PC2']-c2['PC2'])**2 + (c1_songs['PC3']-c2['PC3'])**2)
                                best_c1_match = c1_songs.loc[c1_songs['dist_to_c2'].idxmin()]
                                
                                # Song from c2 closest to c1's centroid
                                c2_songs['dist_to_c1'] = np.sqrt((c2_songs['PC1']-c1['PC1'])**2 + (c2_songs['PC2']-c1['PC2'])**2 + (c2_songs['PC3']-c1['PC3'])**2)
                                best_c2_match = c2_songs.loc[c2_songs['dist_to_c1'].idxmin()]
                                
                                stats_text.append(html.P([
                                    html.Span(f"🤝 Bridge songs:", style={'fontWeight': 'bold', 'color': '#6c757d'}),
                                    html.Br(),
                                    html.Span(f"• {c1['Contributor']}'s pick: ", style={'fontSize': '12px', 'marginLeft': '20px'}),
                                    html.A(f"{best_c1_match['Song']} - {best_c1_match['Artist']}", 
                                           href=f"https://open.spotify.com/track/{best_c1_match['Spotify Track Id']}", 
                                           target="_blank", className='spotify-link'),
                                    html.Br(),
                                    html.Span(f"• {c2['Contributor']}'s pick: ", style={'fontSize': '12px', 'marginLeft': '20px'}),
                                    html.A(f"{best_c2_match['Song']} - {best_c2_match['Artist']}", 
                                           href=f"https://open.spotify.com/track/{best_c2_match['Spotify Track Id']}", 
                                           target="_blank", className='spotify-link')
                                ], style={'fontSize': '12px', 'marginLeft': '20px'}))
            
            # --- Overall Musical Taste Extremes ---
            # If no contributor AND no genre filter is applied...
            if not selected_contributors and not selected_genres and len(centroids) > 2:
                all_distances = []
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        c1 = centroids[i]
                        c2 = centroids[j]
                        d = np.sqrt((c1['PC1']-c2['PC1'])**2 + (c1['PC2']-c2['PC2'])**2 + (c1['PC3']-c2['PC3'])**2)
                        all_distances.append({'pair': (c1['Contributor'], c2['Contributor']), 'dist': d})
                
                if all_distances:
                    most_similar = min(all_distances, key=lambda x: x['dist'])
                    most_dissimilar = max(all_distances, key=lambda x: x['dist'])
                    
                    stats_text.append(html.P([
                        html.Span("Most Similar: ", style={'fontWeight': 'bold'}),
                        f"{most_similar['pair'][0]} & {most_similar['pair'][1]} ({100 / (1 + most_similar['dist']):.1f}% similarity)"
                    ]))
                    stats_text.append(html.P([
                        html.Span("Most Dissimilar: ", style={'fontWeight': 'bold'}),
                        f"{most_dissimilar['pair'][0]} & {most_dissimilar['pair'][1]} ({100 / (1 + most_dissimilar['dist']):.1f}% similarity)"
                    ]))

                    # Variety / Cohesion based on Variance
                    cont_vars = []
                    for cont in plot_df['Contributor'].unique():
                        c_data = plot_df[plot_df['Contributor'] == cont]
                        if len(c_data) > 2: # Need enough points for meaningful variance
                            v = c_data[['PC1', 'PC2', 'PC3']].var().sum()
                            cont_vars.append({'cont': cont, 'var': v})
                    
                    if cont_vars:
                        # Sort for ranking (Most Diverse to Most Narrow)
                        cont_vars.sort(key=lambda x: x['var'], reverse=True)
                        
                        most_diverse = cont_vars[0]
                        most_cohesive = cont_vars[-1]
                        
                        stats_text.append(html.P([
                            html.Span("Most Narrow Taste: ", style={'fontWeight': 'bold'}),
                            f"{most_cohesive['cont']}"
                        ]))
                        stats_text.append(html.P([
                            html.Span("Most Diverse Taste: ", style={'fontWeight': 'bold'}),
                            f"{most_diverse['cont']}"
                        ]))

                        # Full Ranking Dropdown
                        ranking_items = [html.Li(f"{cv['cont']}") for cv in cont_vars]
                        stats_text.append(html.Details([
                            html.Summary("Full Ranking of Music Taste Diversity 🌎", style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginTop': '5px', 'color': '#6c757d'}),
                            html.Ol(ranking_items, style={'fontSize': '12px', 'marginTop': '10px', 'paddingLeft': '20px'})
                        ]))

                    # Underground Ranking based on Average Popularity
                    cont_pop = []
                    for cont in plot_df['Contributor'].unique():
                        c_data = plot_df[plot_df['Contributor'] == cont]
                        if not c_data.empty:
                            avg_pop = c_data['Popularity'].mean()
                            cont_pop.append({'cont': cont, 'pop': avg_pop})
                    
                    if cont_pop:
                        # Sort for ranking (Lowest Popularity = Most Underground)
                        cont_pop.sort(key=lambda x: x['pop'])
                        
                        most_underground = cont_pop[0]
                        
                        stats_text.append(html.P([
                            html.Span("Most Underground: ", style={'fontWeight': 'bold'}),
                            f"{most_underground['cont']} (avg pop: {most_underground['pop']:.1f})"
                        ]))

                        # Full Ranking Dropdown
                        underground_ranking_items = [html.Li(f"{cp['cont']} ({cp['pop']:.1f})") for cp in cont_pop]
                        stats_text.append(html.Details([
                            html.Summary("Full Ranking of Underground Taste 💎", style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginTop': '5px', 'color': '#6c757d'}),
                            html.Ol(underground_ranking_items, style={'fontSize': '12px', 'marginTop': '10px', 'paddingLeft': '20px'})
                        ]))

                # --- Agreeable vs Polarizing Songs ---
                # Calculate distance of every song to the global centroid
                g_x = plot_df['PC1'].mean()
                g_y = plot_df['PC2'].mean()
                g_z = plot_df['PC3'].mean()
                
                plot_df['dist_to_global'] = np.sqrt(
                    (plot_df['PC1'] - g_x)**2 + 
                    (plot_df['PC2'] - g_y)**2 + 
                    (plot_df['PC3'] - g_z)**2
                )
                
                agreeable = plot_df.loc[plot_df['dist_to_global'].idxmin()]
                polarizing = plot_df.loc[plot_df['dist_to_global'].idxmax()]
                
                stats_text.append(html.P([
                    html.Span("Most Agreeable: ", style={'fontWeight': 'bold', 'color': '#28a745'}),
                    html.A(f"{agreeable['Song']} - {agreeable['Artist']}", 
                           href=f"https://open.spotify.com/track/{agreeable['Spotify Track Id']}", 
                           target="_blank", className='spotify-link'),
                    f" ({agreeable['Contributor']})"
                ], style={'fontSize': '16px'}))
                stats_text.append(html.P([
                    html.Span("Most Divergent: ", style={'fontWeight': 'bold', 'color': '#dc3545'}),
                    html.A(f"{polarizing['Song']} - {polarizing['Artist']}", 
                           href=f"https://open.spotify.com/track/{polarizing['Spotify Track Id']}", 
                           target="_blank", className='spotify-link'),
                    f" ({polarizing['Contributor']})"
                ], style={'fontSize': '16px'}))

    # --- Search Highlight ---
    if searched_song_id:
        highlight_row = df_clean[df_clean['Spotify Track Id'] == searched_song_id].copy()
        if not highlight_row.empty:
            highlight_trace = px.scatter_3d(
                highlight_row,
                x='PC1', y='PC2', z='PC3',
                hover_data=hover_cols
            ).data[0]
            highlight_trace.marker.size = 20
            highlight_trace.marker.opacity = 1.0
            highlight_trace.marker.color = 'yellow'
            highlight_trace.marker.line = dict(width=2, color='black')
            highlight_trace.name = 'Searched Song'
            fig.add_trace(highlight_trace)
            stats_text.insert(0, html.P("Song highlighted in yellow.", style={'color': '#d4a017', 'fontWeight': 'bold'}))
            
            # Find closest song from the matched list
            if not matched_df.empty:
                s_x, s_y, s_z = highlight_row.iloc[0]['PC1'], highlight_row.iloc[0]['PC2'], highlight_row.iloc[0]['PC3']
                
                # Exclude the searched song itself from the matched list for comparison
                others_in_match = matched_df[matched_df['Spotify Track Id'] != searched_song_id].copy()
                
                if not others_in_match.empty:
                    others_in_match['dist_to_search'] = np.sqrt(
                        (others_in_match['PC1'] - s_x)**2 + 
                        (others_in_match['PC2'] - s_y)**2 + 
                        (others_in_match['PC3'] - s_z)**2
                    )
                    closest_song = others_in_match.loc[others_in_match['dist_to_search'].idxmin()]
                    
                    stats_text.insert(1, html.P([
                        html.Span("Most Similar Song Displayed 👥: ", style={'fontWeight': 'bold'}),
                        html.A(f"{closest_song['Song']} - {closest_song['Artist']}", 
                               href=f"https://open.spotify.com/track/{closest_song['Spotify Track Id']}", 
                               target="_blank", className='spotify-link'),
                        f" ({closest_song['Contributor']})"
                    ], style={'fontSize': '14px', 'color': '#6c757d', 'marginBottom': '10px'}))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor='#121212' if is_dark else '#ffffff',
        plot_bgcolor='#121212' if is_dark else '#ffffff'
    )

    return fig, stats_text

if __name__ == '__main__':
    # Running on 8050 by default
    app.run()
