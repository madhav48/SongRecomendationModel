# Spotify Song Recommendation Player

An application for recommending songs and controlling Spotify playback using the Spotify Web API.

---

## 1. Usage

### Prerequisites

- Python 3.10+
- A [Spotify Developer Account](https://developer.spotify.com/)
- Spotify Premium account (required for playback control)

### Setup Steps

1. **Clone the Repository**

   ```sh
   git clone git@github.com:madhav48/SongRecomendationModel.git
   cd SongRecomendationModel
   ```

2. **Create a Spotify Developer App**

   - Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
   - Click "Create an App".
   - Note your **Client ID** and **Client Secret**.
   - Set the Redirect URI (e.g., `http://localhost:8888/callback`) in your app settings.

3. **Configure the `.env` File**

   Create a `.env` file in the project root with the following content:

   ```
   SPOTIPY_CLIENT_ID=your_client_id
   SPOTIPY_CLIENT_SECRET=your_client_secret
   SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
   ```

4. **Install Dependencies**

   ```sh
   pip install spotipy pandas scikit-learn
   ```

5. **Run the Application**

   - To start the main program, run:

     ```sh
     python main.py
     ```

   - Follow the prompts to log in to Spotify and use the recommendation features.

---

## 2. File Directory

```
.
├── .env                       # Environment variables for Spotify API credentials
├── main.py                    # Main entry point for the application
├── recommend_songs.py         # Song recommendation logic and ML model interface
├── spotify_player.py          # Spotify playback control and integration
├── assets/
    ├── clustering_model.pkl       # Pre-trained clustering model for recommendations
    ├── song_feat_data.csv        # Song features dataset used for recommendations
    └── spotify_tracks_data.csv   # Spotify tracks metadata
```
---

## Additional Notes

- Ensure you have an active Spotify device (e.g., Spotify app open and playing) for playback features to work.
- For more details on the recommendation algorithm, see comments in `recommend_songs.py`.
- For troubleshooting Spotify authentication, refer to the [Spotipy documentation](https://spotipy.readthedocs.io/).

---
