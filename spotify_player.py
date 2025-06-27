import spotipy
from spotipy.oauth2 import SpotifyOAuth
import threading
import time
from recommend_songs import SongRecommender


class SpotifyPlayer:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.scope = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=self.scope
        ))
        self.recommender = SongRecommender()
        self.recommender.played_titles.append(self.sp.current_playback()['item']['name'])


    def get_active_device(self):
        devices = self.sp.devices().get('devices', [])
        if not devices:
            print("No active device found. Open Spotify and play a song.")
            return None
        return devices[0]['id']

    def get_current_song(self):
        current = self.sp.current_playback()
        if current and current.get('item'):
            track = current['item']
            return {
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'uri': track['uri']
            }
        return None

    def play_song(self, uri):
        device_id = self.get_active_device()
        if device_id:
            self.sp.start_playback(device_id=device_id, uris=[uri])

    def pause(self):
        device_id = self.get_active_device()
        if device_id:
            self.sp.pause_playback(device_id=device_id)

    def resume(self):
        device_id = self.get_active_device()
        if device_id:
            self.sp.start_playback(device_id=device_id)

    def next_track(self):
        device_id = self.get_active_device()
        if device_id:
            self.sp.next_track(device_id=device_id)

    def previous_track(self):
        device_id = self.get_active_device()
        if device_id:
            self.sp.previous_track(device_id=device_id)

    def search_and_play(self, track_name, artist_name=None):
        query = f'track:{track_name}'
        if artist_name:
            query += f' artist:{artist_name}'

        print(f"Searching with query: {query}")
        try:
            result = self.sp.search(q=query, type='track', limit=1)
            tracks = result.get('tracks', {}).get('items', [])

            if not tracks:
                print("No exact track match found.")
                return

            track = tracks[0]
            uri = track['uri']
            name = track['name']
            artist = track['artists'][0]['name']

            device_id = self.get_active_device()
            if not device_id:
                print("No active device found.")
                return

            self.sp.start_playback(device_id=device_id, uris=[uri])
            print(f"Now Playing: {name} by {artist}")

        except Exception as e:
            print(f"Error during search/play: {e}")

    def get_playback_state(self):
        return self.sp.current_playback()
    

    def add_to_queue(self, uri):
        try:
            self.sp.add_to_queue(uri)
            print(f"Queued: {uri}")
        except Exception as e:
            print(f"Failed to queue song: {e}")

    def start_auto_queue(self, poll_interval=5):
        """
        Continuously checks the currently playing song.
        When the song changes, calls the song recommender to get `n` next URIs and queues them all.
        """

        def auto_queue_loop():
            print("Auto-queue started.")
            last_track_id = None

            while True:
                playback = self.sp.current_playback()
                if playback and playback.get('item'):
                    current_id = playback['item']['id']

                    if not last_track_id or current_id != last_track_id:
                        recommended_uris = self.recommender.predict_next_songs(current_id)
                        if recommended_uris:
                            for uri in recommended_uris:
                                self.add_to_queue(uri)

                    last_track_id = current_id

                time.sleep(poll_interval)

        # Run in background thread
        threading.Thread(target=auto_queue_loop, daemon=True).start()
