import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def multi_value_one_hot(df, column):
    # Crea dummies dalle stringhe separate da virgole
    s = df[column].str.get_dummies(sep=', ')

    # Per ogni colonna generata, aggiungi o aggiorna i valori nel DataFrame originale
    for col in s.columns:
        prefixed_col = column + '_' + col
        if prefixed_col in df.columns:
            # Se la colonna esiste già, aggiorna i valori
            df[prefixed_col] = df[prefixed_col] | s[col]
        else:
            # Altrimenti, crea una nuova colonna
            df[prefixed_col] = s[col]

    return df


class DropOtherColumns(BaseEstimator, TransformerMixin):
    def __init__(self, prefix='Other'):
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Trova i nomi delle colonne che iniziano con il prefisso specificato
        other_columns = [col for col in X.columns if col.startswith(self.prefix)]
        # Rimuovi le colonne trovate
        return X.drop(columns=other_columns, errors='ignore')

# Carica il classificatore
with open('models/classification/AdaBoostClassifier.pkl', 'rb') as file:
    classifier=pickle.load(file)

# Carica il file CSV con i premi
awards_df = pd.read_csv('Prizes_database/nominee_total_nominations_with_roles.csv')

# Funzione per recuperare i premi basati sui nomi
def get_awards_info(actors, director, writers):
    awards_info = {
        'cast_globe_nomination': 0,
        'dir_oscar_nomination': 0,
        'writer_oscar_nomination': 0,
        'BAFTA_act_nom': 0,
        'BAFTA_dir_nom': 0,
        'BAFTA_writer_nom': 0,
        'dir_emmy_nom': 0,
        'writer_emmy_nom': 0,
        'act_emmy_nom': 0,
        'actors_films_before': 0,
        'director_films_before': 0,
        'writers_films_before': 0
    }

    for actor in actors:
        actor_awards = awards_df[awards_df['nominee'] == actor]
        if not actor_awards.empty:
            awards_info['cast_globe_nomination'] += actor_awards['golden_globe_nomination'].values[0]
            awards_info['BAFTA_act_nom'] += actor_awards['bafta_nomination'].values[0]
            awards_info['act_emmy_nom'] += actor_awards['emmy_nomination'].values[0]
            awards_info['actors_films_before'] += actor_awards['actor_count'].values[0]

    director_awards = awards_df[awards_df['nominee'] == director]
    if not director_awards.empty:
        awards_info['dir_oscar_nomination'] = director_awards['oscar_nomination'].values[0]
        awards_info['BAFTA_dir_nom'] = director_awards['bafta_nomination'].values[0]
        awards_info['dir_emmy_nom'] = director_awards['emmy_nomination'].values[0]
        awards_info['director_films_before'] = director_awards['director_count'].values[0]

    for writer in writers:
        writer_awards = awards_df[awards_df['nominee'] == writer]
        if not writer_awards.empty:
            awards_info['writer_oscar_nomination'] += writer_awards['oscar_nomination'].values[0]
            awards_info['BAFTA_writer_nom'] += writer_awards['bafta_nomination'].values[0]
            awards_info['writer_emmy_nom'] += writer_awards['emmy_nomination'].values[0]
            awards_info['writers_films_before'] += writer_awards['writer_count'].values[0]

    return awards_info

# Funzione per fare una predizione
def predict_revenue_cluster():
    try:
        duration = int(duration_entry.get())
        converted_budget = float(budget_entry.get())
        actors = actor_entry.get().split(',')
        director = director_entry.get()
        writers = writer_entry.get().split(',')
        genre = genre_entry.get()
        language = language_entry.get()
        production_company = production_company_entry.get()
        month_published = int(month_entry.get())

        awards_info = get_awards_info(actors, director, writers)

        data = pd.DataFrame({
            'duration': [duration],
            'converted_budget': [converted_budget],
            'genre': [genre],
            'language': [language],
            'production_company': [production_company],
            'month_published': [month_published],
            'cast_globe_nomination': [awards_info['cast_globe_nomination']],
            'dir_oscar_nomination': [awards_info['dir_oscar_nomination']],
            'writer_oscar_nomination': [awards_info['writer_oscar_nomination']],
            'BAFTA_act_nom': [awards_info['BAFTA_act_nom']],
            'BAFTA_dir_nom': [awards_info['BAFTA_dir_nom']],
            'BAFTA_writer_nom': [awards_info['BAFTA_writer_nom']],
            'dir_emmy_nom': [awards_info['dir_emmy_nom']],
            'writer_emmy_nom': [awards_info['writer_emmy_nom']],
            'act_emmy_nom': [awards_info['act_emmy_nom']],
            'actors_films_before': [awards_info['actors_films_before']],
            'director_films_before': [awards_info['director_films_before']],
            'writers_films_before': [awards_info['writers_films_before']],
            'genre_Action':0,
            'genre_Adult':0,
            'genre_Adventure':0,
            'genre_Animation':0,
            'genre_Biography':0,
            'genre_Comedy':0,
            'genre_Crime':0,
            'genre_Documentary':0,
            'genre_Drama':0,
            'genre_Family':0,
            'genre_Fantasy':0,
            'genre_Film-Noir':0,
            'genre_History':0,
            'genre_Horror':0,
            'genre_Music':0,
            'genre_Musical':0,
            'genre_Mystery':0,
            'genre_Romance':0,
            'genre_Sci-Fi':0,
            'genre_Sport':0,
            'genre_Thriller':0,
            'genre_War':0,
            'genre_Western':0

        })

        data = multi_value_one_hot(data, 'genre')
        data.drop(columns=['genre'], inplace=True)
        data['month_published'] = data['month_published'].astype(str)

        prediction = classifier.predict(data)
        messagebox.showinfo("Predizione", f"Il cluster di revenue predetto è: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Errore", str(e))

# Creazione della finestra GUI
root = tk.Tk()
root.title("Predizione del Revenue Cluster")

# Creazione dei campi di input
tk.Label(root, text="Duration").grid(row=0)
duration_entry = tk.Entry(root)
duration_entry.grid(row=0, column=1)

tk.Label(root, text="Converted Budget").grid(row=1)
budget_entry = tk.Entry(root)
budget_entry.grid(row=1, column=1)

tk.Label(root, text="Actors (comma separated)").grid(row=2)
actor_entry = tk.Entry(root)
actor_entry.grid(row=2, column=1)

tk.Label(root, text="Director").grid(row=3)
director_entry = tk.Entry(root)
director_entry.grid(row=3, column=1)

tk.Label(root, text="Writers (comma separated)").grid(row=4)
writer_entry = tk.Entry(root)
writer_entry.grid(row=4, column=1)

tk.Label(root, text="Genre").grid(row=5)
genre_entry = tk.Entry(root)
genre_entry.grid(row=5, column=1)

tk.Label(root, text="Language").grid(row=6)
language_entry = tk.Entry(root)
language_entry.grid(row=6, column=1)

tk.Label(root, text="Production Company").grid(row=7)
production_company_entry = tk.Entry(root)
production_company_entry.grid(row=7, column=1)

tk.Label(root, text="Month Published").grid(row=8)
month_entry = tk.Entry(root)
month_entry.grid(row=8, column=1)

# Bottone per fare la predizione
predict_button = tk.Button(root, text="Predici", command=predict_revenue_cluster)
predict_button.grid(row=9, columnspan=2)

# Avvio della GUI
root.mainloop()
