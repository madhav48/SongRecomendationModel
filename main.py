import time
from spotify_player import SpotifyPlayer
from dotenv import load_dotenv
import os


load_dotenv()

player = SpotifyPlayer(
 client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    redirect_uri=os.getenv("REDIRECT_URI"),
)

player.start_auto_queue()

while True:
    time.sleep(1)  # Keep the script running to allow auto-queueing


