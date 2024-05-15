import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances

features = ['valence', 'acousticness','year',
            'danceability', 'duration_ms', 'energy',
            'explicit','instrumentalness', 'key', 
            'liveness', 'loudness', 'mode',
            'popularity','speechiness', 'tempo']

metadata_cols = ['release_date', 'name',  'artists']

tracks = pd.read_csv('data/tracks.csv')
tracks['year'] = tracks.apply(lambda row: int(row['release_date'][:4]), axis=1)

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=8, 
                                   verbose=2))],verbose=True)   


song_cluster_pipeline.fit(tracks[features])

def input_preprocessor(song_list, dataset):
    song_vectors = []
    for song in song_list:
        try:
            song_data = dataset[(dataset['name'] == song['name']) &
                                (dataset['release_date'] == song['release_date'])].iloc[0]
        except IndexError:
            song_data = None
        if song_data is None:
            print('Warning: {} does not exist in our database'.format(song['name']))
            continue
        song_vectors.append(song_data[features].values) 
        if not song_vectors:
            raise ValueError("None of the songs in the song list exist in the dataset.")
    return np.mean(np.array(list(song_vectors)), axis=0)


def Music_Recommender(song_list, dataset, n_songs=10):
    song_center = input_preprocessor(song_list, dataset)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(dataset[features])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))    
    ed_dist = euclidean_distances(scaled_song_center, scaled_data)
    index = list(np.argsort(ed_dist)[:,:n_songs][0])
    rec_output = dataset.iloc[index]
    return rec_output[metadata_cols]


results =Music_Recommender([{'name': 'Toosie Slide', 'release_date': '2020-04-03'},
                                          {'name': 'Outta Time (feat. Drake)', 'release_date': '2020-10-02'},
                                          {'name': 'Chicago Freestyle (feat. Giveon)', 'release_date': '2020-05-01'}],tracks, 10)
print(results.head())