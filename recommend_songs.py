import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
import os
from rapidfuzz.fuzz import token_sort_ratio



class SongRecommender:
    def __init__(self):
        """
        Initialize the SongRecommender with pre-trained KMedoids and PCA data.
        """
        assets_dir = os.path.abspath("./assets")
        model_path = os.path.join(assets_dir, 'clustering_model.pkl')
        pca_csv_path = os.path.join(assets_dir, 'song_feat_data.csv')

        # Load KMedoids model
        with open(model_path, 'rb') as f:
            self.kmeans = pickle.load(f)

        # Load PCA and song metadata
        self.df = pd.read_csv(pca_csv_path)

        # Split PCA components and original features
        self.X_pca = self.df[[f'PC{i}' for i in range(1, 9)]].values
        self.X_df = self.df.drop(columns=['track_name', 'track_id'] + [f'PC{i}' for i in range(1, 9)])

        # Store track names and IDs
        self.track_names = self.df['track_name'].tolist()
        self.track_ids = self.df['track_id'].tolist()
        self.labels = self.kmeans.labels_
        self.played_titles = []  


    def predict_next_songs(self, track_id: str, n: int = 3, max_tries: int = 10):
        """
        Return up to n similar song track_ids for a given input track_id.
        Adds randomness among top-n suggestions.
        Avoids recently played songs, but falls back if needed.
        """
        if track_id not in self.df['track_id'].values:
            print(f"Track ID '{track_id}' not found in the dataset.")
            return []
        
        # Add current song to history if not already present.
        if track_id not in self.played_titles:
            current_song_name = self.df[self.df['track_id'] == track_id]['track_name'].values[0]
            self.played_titles.append(current_song_name)

        song_index = self.df[self.df['track_id'] == track_id].index[0]
        song_data_pca = self.X_pca[song_index]
        song_data_feat = self.X_df.iloc[song_index].values

        tries = 0
        batch_size = n
        offset = 0
        filtered = []

        while tries < max_tries:
            closest_song_indices = self._find_closest_songs(
                song_data_pca, song_data_feat, song_index, n_neighbors=batch_size + offset
            )

            batch_indices = closest_song_indices[offset:offset + batch_size]
            recommended_tracks = self.df.iloc[batch_indices].copy()
            recommended_tracks = recommended_tracks.sample(frac=1).reset_index(drop=True)

            filtered = [
                row for _, row in recommended_tracks.iterrows()
                if not self.is_duplicate(row['track_name'], self.played_titles)
            ]
            if filtered:
                break
            offset += batch_size
            tries += 1

        # If still no non-duplicates, clear history and try again with the first batch
        if not filtered:
            print("No non-duplicate songs found after retries. Clearing history and retrying.")
            self.played_titles.clear()
            closest_song_indices = self._find_closest_songs(
                song_data_pca, song_data_feat, song_index, n_neighbors=batch_size
            )
            recommended_tracks = self.df.iloc[closest_song_indices].copy()
            recommended_tracks = recommended_tracks.sample(frac=1).reset_index(drop=True)
            filtered = [
                row for _, row in recommended_tracks.iterrows()
                if not self.is_duplicate(row['track_name'], self.played_titles)
            ]

        # Still empty -> Allow duplicates (fallback)
        if not filtered:
            print("No unique songs found even after clearing history. Using duplicates.")
            filtered = recommended_tracks.to_dict('records')

        # Use first from final filtered list
        if filtered:
            next_song = filtered[0]
            print(f"Next song selected: {next_song['track_name']}")
            self.played_titles.append(next_song['track_name'])
            return [next_song['track_id']]
        else:
            print("Unexpected issue: still no song selected.")
            return []

        

    def _find_closest_songs(self, song_data_pca, song_data_feat, song_index, n_neighbors=5, metric='cosine'):
        """
        Find n closest songs within the same cluster.
        """
        song_cluster = self.kmeans.predict([song_data_pca])[0]
        cluster_mask = self.labels == song_cluster
        cluster_indices = np.where(cluster_mask)[0]

        cluster_points = self.X_df.iloc[cluster_mask].values
        distances = pairwise_distances([song_data_feat], cluster_points, metric=metric).flatten()

        # Exclude the song itself
        is_self = np.all(np.isclose(cluster_points, song_data_feat), axis=1)
        distances[is_self] = np.inf

        # Sort by distance and get top N
        sorted_idx_within_cluster = np.argsort(distances)[:n_neighbors]
        closest_df_indices = cluster_indices[sorted_idx_within_cluster]

        return closest_df_indices.tolist()
    

    def is_duplicate(self, new_title: str, history_titles: list, threshold: int = 70):
        for old_title in history_titles:
            similarity = token_sort_ratio(new_title.lower(), old_title.lower())
            # print(f"'{new_title}' is similar to '{old_title}' with similarity {similarity}.")
            if similarity >= threshold:
                return True
        return False
