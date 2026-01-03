import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def create_evolution_video():
    # Load data
    file_path = '/Users/tjvitchutripop/Documents/projects/chancla-mix-analysis/pca_results.csv'
    df = pd.read_csv(file_path)
    
    # Convert 'Added At' to datetime and sort
    df['Added At'] = pd.to_datetime(df['Added At'])
    df = df.sort_values('Added At').reset_index(drop=True)
    
    # Setup aesthetics
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    
    # Contributor colors
    contributors = df['Contributor'].unique()
    colors = plt.cm.get_cmap('hsv', len(contributors))
    color_map = {contributor: colors(i) for i, contributor in enumerate(contributors)}
    
    # Axis limits
    padding = 1.0
    x_min, x_max = df['PC1'].min() - padding, df['PC1'].max() + padding
    y_min, y_max = df['PC2'].min() - padding, df['PC2'].max() + padding
    z_min, z_max = df['PC3'].min() - padding, df['PC3'].max() + padding
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel('PC1', color='white', alpha=0.6)
    ax.set_ylabel('PC2', color='white', alpha=0.6)
    ax.set_zlabel('PC3', color='white', alpha=0.6)
    
    # Hide panes for a cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#111111')
    ax.yaxis.pane.set_edgecolor('#111111')
    ax.zaxis.pane.set_edgecolor('#111111')
    
    ax.grid(True, linestyle='--', alpha=0.1)
    
    # Title and labels (using fig text for fixed position)
    title_text = fig.text(0.5, 0.95, '', ha='center', fontsize=16, color='white', fontweight='bold')
    song_text = fig.text(0.5, 0.05, '', ha='center', fontsize=11, color='#1DB954', fontweight='medium', wrap=True)
    
    # Scatter objects
    scatters = {contributor: ax.scatter([], [], [], c=[color_map[contributor]], label=contributor, s=40, alpha=0.7, edgecolors='none') 
                for contributor in contributors}
    
    # Latest point marker
    latest_marker, = ax.plot([], [], [], 'wo', markersize=10, markeredgecolor='white', markerfacecolor='none', markeredgewidth=2)
    
    legend = ax.legend(loc='upper right', frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color("white")

    def init():
        for s in scatters.values():
            s._offsets3d = (np.array([]), np.array([]), np.array([]))
        latest_marker.set_data([], [])
        latest_marker.set_3d_properties([])
        title_text.set_text('Chancla Mix Evolution (3D)')
        song_text.set_text('')
        return list(scatters.values()) + [latest_marker]

    def update(frame):
        current_data = df.iloc[:frame+1]
        last_song = df.iloc[frame]
        
        # Update each contributor's scatter
        for contributor in contributors:
            contrib_data = current_data[current_data['Contributor'] == contributor]
            if not contrib_data.empty:
                scatters[contributor]._offsets3d = (
                    contrib_data['PC1'].values,
                    contrib_data['PC2'].values,
                    contrib_data['PC3'].values
                )
        
        # Highlight the latest song
        latest_marker.set_data([last_song['PC1']], [last_song['PC2']])
        latest_marker.set_3d_properties([last_song['PC3']])
        
        # Rotate view slowly
        ax.view_init(elev=20, azim=frame * 0.5)
        
        # Update text
        date_str = last_song['Added At'].strftime('%Y-%m-%d')
        title_text.set_text(f'Playlist Evolution: {date_str}')
        
        # Truncate song title if too long
        song_info = f"{last_song['Song']} - {last_song['Artist']} ({last_song['Contributor']})"
        song_text.set_text(song_info)
        
        return list(scatters.values()) + [latest_marker]

    # Create animation
    print("Generating 3D animation frames...")
    # Blit=False is often safer for 3D axis rotation
    ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=False, interval=100)
    
    # Save video
    output_file = '/Users/tjvitchutripop/Documents/projects/chancla-mix-analysis/playlist_evolution_3d.mp4'
    print(f"Saving video to {output_file}...")
    writer = FFMpegWriter(fps=15, bitrate=2500)
    ani.save(output_file, writer=writer)
    print("Done!")

if __name__ == "__main__":
    create_evolution_video()
