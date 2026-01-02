import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
PLAYLIST_ID = os.getenv('SPOTIFY_PLAYLIST_ID')

def get_playlist_tracks(sp, playlist_id):
    """
    Fetches all tracks from a Spotify playlist, handling pagination.
    """
    results = sp.playlist_items(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

id_to_name = {
    "7i1au5r0wt5g04sdi31f2wp1j": "Miller",
    "bill_clint0n": "Shelbie",
    "ozs7w2h8ep9kwl8lk9p5mhbyk": "Puffy",
    "21idy57p2a3mk26lblqlsjlzy": "TJ",
    "09b91lg770ielooktatymt7h6": "Angel",
    "19inrr11c14y9mv974v9ke5k0": "Gabe",
    "91awgkdb6b47oq1e3vbn3fea5": "Sarah",
    "31mtw36aec2ylubx624ehhiazgcu": "Clark",
    "wzrdonkey": "Tony",
    "lolab7": "Jean",
    "7gz49zeml7gld9xqt8xjiso8i": "Micah",
    "31uhpgiuzyhav44cn4gcthk4bin4": "Bryce",
    "peachylolita": "Theresa",
    "whoskiba": "Hannah",
    "loki_poki13": "Kathryn",
    "33125fd5seh9g4nj44rns04pj": "Sam",
    "guesswhoswaggy": "Eunice",
    "nljwehfpnp8a8xw1a8bcmo6uh": "Aiden",
    "bgpd96kra5rcfkmrzgy0pt69c": "Nora",
}

def main():
    if not CLIENT_ID or not CLIENT_SECRET or not PLAYLIST_ID:
        print("Error: Please check your .env file for Spotify credentials and Playlist ID.")
        return

    # Authenticate
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    print(f"Fetching tracks for playlist ID: {PLAYLIST_ID}...")
    try:
        raw_tracks = get_playlist_tracks(sp, PLAYLIST_ID)
    except Exception as e:
        print(f"Error fetching playlist: {e}")
        return

    print(f"Found {len(raw_tracks)} tracks. Processing...")

    final_data = []
    
    # Process tracks one by one or in small chunks if we were using the old API
    # Now we iterate and fetch features from Reccobeats
    
    total_tracks = len(raw_tracks)
    for index, item in enumerate(raw_tracks):
        if not item['track']:
            continue
            
        track = item['track']
        track_id = track['id']
        added_by = item.get('added_by', {})
        contributor_id = added_by.get('id', 'Unknown')
        
        print(f"Processing {index + 1}/{total_tracks}: {track['name']}")
        
        track_info = {
            'Track Name': track['name'],
            'Artist': ", ".join([artist['name'] for artist in track['artists']]),
            'Contributed By': id_to_name[contributor_id],
            'Spotify ID': track_id
        }
        
        final_data.append(track_info)

    # Create DataFrame
    df = pd.DataFrame(final_data)
    
    output_file = 'playlist_contributor_identifier.csv'
    df.to_csv(output_file, index=False)
    print(f"Done! Data saved to {output_file}")

    # Add contributor column to existing data in data.csv
    df2 = pd.read_csv('data.csv')
    # Use Spotify ID to match to song in df
    df2['Contributor'] = df2['Spotify Track Id'].map(df.set_index('Spotify ID')['Contributed By'])
    df2.to_csv('data_new.csv', index=False)
    print("Done! Data saved to data_new.csv")

if __name__ == "__main__":
    main()

    
