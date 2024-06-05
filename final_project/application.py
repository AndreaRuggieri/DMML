import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Caricamento dei dati per i menu a tendina
actors_df = pd.read_csv('Prizes_database/filtered_prizes.csv')
awards_df = pd.read_csv('Prizes_database/filtered_prizes.csv')
writers_df = pd.read_csv('writers.csv')
directors_df = pd.read_csv('directors.csv')
genres_df = pd.read_csv('genres.csv')
production_companies_df = pd.read_csv('production_companies.csv')
languages_df = pd.read_csv('languages.csv')

actor_list = actors_df['nominee'].dropna().unique().tolist()
writer_list = writers_df['writer'].dropna().unique().tolist()
director_list = directors_df['director'].dropna().unique().tolist()
genre_list = genres_df['genre'].dropna().unique().tolist()
company_list = production_companies_df['company'].dropna().unique().tolist()
language_list = languages_df['language'].dropna().unique().tolist()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

def one_hot_encode_and_replace(df, column):
    # Crea dummies dalla colonna categoriale
    dummies = pd.get_dummies(df[column], prefix=column)

    # Unisci i dummies al DataFrame originale, sostituendo la colonna esistente
    for col in dummies.columns:
        df[col] = dummies[col]

    # Rimuovi la colonna originale
    df.drop(columns=[column], inplace=True)

    return df

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

with open('models/regression/RandomForestRegressor.pkl', 'rb') as file:
    regressor=pickle.load(file)

class DynamicFieldsApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Predizione del Revenue Cluster")
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        main_frame = ttk.Frame(master)
        main_frame.grid(sticky='nsew')

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)

        self.actor_frames = []
        self.writer_frames = []
        self.genre_frames = []
        # Carica l'immagine
        self.load_image(main_frame)

        # Actor fields
        ttk.Label(main_frame, text="Insert actor", anchor='w').grid(row=0, column=0, sticky='W')
        self.add_actor_field(main_frame, removable=False, index=1)

        # Writer fields
        ttk.Label(main_frame, text="Insert writer", anchor='w').grid(row=200, column=0, sticky='W')
        self.add_writer_field(main_frame, removable=False, index=201)

        ttk.Label(main_frame, text="Insert genre", anchor='w').grid(row=300, column=0, sticky='W')
        self.add_genre_field(main_frame, removable=False, index=301)

        # Director field
        ttk.Label(main_frame, text="Insert director", anchor='w').grid(row=400, column=0, sticky='W')
        self.director_cb = ttk.Combobox(main_frame, values=director_list, state="readonly")
        self.director_cb.grid(row=400, column=1, sticky='W')
        self.director_cb.bind("<<ComboboxSelected>>", self.check_conditions)

        # Initialize the labels for duration and budget
        self.duration_label = ttk.Label(main_frame, text="120")
        self.budget_label = ttk.Label(main_frame, text="10,000,000")

        self.setup_other_fields(main_frame)
        self.update_predict_button_state()

    def load_image(self, parent):
        # Carica l'immagine e posizionala nella finestra
        image = Image.open('image.png')
        image = image.resize((400, 600))
        self.photo = ImageTk.PhotoImage(image)
        label = ttk.Label(parent, image=self.photo)
        label.grid(row=0, column=3, rowspan=1100, padx=20, pady=20, sticky='nsew')

    def setup_other_fields(self, parent):
        # Duration Slider
        ttk.Label(parent, text="Duration").grid(row=500, sticky='W')
        self.duration_values = list(range(50, 251, 10))
        self.duration_scale = ttk.Scale(parent, from_=0, to=len(self.duration_values) - 1, orient='horizontal',
                                        length=200, command=self.update_duration)
        self.duration_scale.set(12)
        self.duration_scale.grid(row=500, column=1, sticky='W')
        self.duration_label.grid(row=500, column=2, sticky='W')  # Positioning the label

        # Budget Slider
        ttk.Label(parent, text="Converted Budget").grid(row=600, sticky='W')
        self.budget_values = list(range(100000, 500000001, 50000))
        self.budget_scale = ttk.Scale(parent, from_=0, to=len(self.budget_values) - 1, orient='horizontal',
                                      length=200, command=self.update_budget)
        self.budget_scale.set(200)
        self.budget_scale.grid(row=600, column=1, sticky='W')
        self.budget_label.grid(row=600, column=2, sticky='W')  # Positioning the label


        # Language Dropdown
        ttk.Label(parent, text="Language").grid(row=800, sticky='W')
        self.language_cb = ttk.Combobox(parent, values=language_list, state="readonly")
        self.language_cb.grid(row=800, column=1, sticky='W')
        self.language_cb.bind("<<ComboboxSelected>>", self.check_conditions)

        # Production Company Dropdown
        ttk.Label(parent, text="Production Company").grid(row=900, sticky='W')
        self.company_cb = ttk.Combobox(parent, values=company_list, state="readonly")
        self.company_cb.grid(row=900, column=1, sticky='W')
        self.company_cb.bind("<<ComboboxSelected>>", self.check_conditions)

        # Month Published Dropdown
        ttk.Label(parent, text="Month Published").grid(row=1000, sticky='W')
        self.month_cb = ttk.Combobox(parent, values=months, state="readonly")
        self.month_cb.grid(row=1000, column=1, sticky='W')

        # Prediction button
        self.predict_button = ttk.Button(parent, text="Predici", command=self.predict_revenue_cluster, state='disabled')
        self.predict_button.grid(row=1100, columnspan=2, pady=10, sticky='W')

    def update_duration(self, val):
        index = int(self.duration_scale.get())
        duration = self.duration_values[index]
        self.duration_label.config(text=str(duration))

    def update_budget(self, val):
        index = int(self.budget_scale.get())
        budget = self.budget_values[index]
        self.budget_label.config(text="{:,}".format(budget))

    def predict_revenue_cluster(self):
        try:
            duration = int(self.duration_label.cget("text"))
            converted_budget = float(self.budget_label.cget("text").replace(",", ""))
            actors = [frame.winfo_children()[0].get() for frame in self.actor_frames]
            director = self.director_cb.get()
            writers = [frame.winfo_children()[0].get() for frame in self.writer_frames]
            genre = [frame.winfo_children()[0].get() for frame in self.genre_frames]
            genre = ', '.join(genre)
            language = self.language_cb.get()
            production_company = self.company_cb.get()
            month_published = str(months.index(self.month_cb.get()) + 1)

            awards_info = self.get_awards_info(actors, director, writers)

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
                'genre_Action': 0,
                'genre_Adult': 0,
                'genre_Adventure': 0,
                'genre_Animation': 0,
                'genre_Biography': 0,
                'genre_Comedy': 0,
                'genre_Crime': 0,
                'genre_Documentary': 0,
                'genre_Drama': 0,
                'genre_Family': 0,
                'genre_Fantasy': 0,
                'genre_Film-Noir': 0,
                'genre_History': 0,
                'genre_Horror': 0,
                'genre_Music': 0,
                'genre_Musical': 0,
                'genre_Mystery': 0,
                'genre_Romance': 0,
                'genre_Sci-Fi': 0,
                'genre_Sport': 0,
                'genre_Thriller': 0,
                'genre_War': 0,
                'genre_Western': 0
            })
            data = multi_value_one_hot(data, 'genre')
            data.drop(columns=['genre'], inplace=True)
            data['month_published'] = data['month_published'].astype(str)

            #regression
            data_reg = pd.DataFrame({
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
                'genre_Action': 0,
                'genre_Adult': 0,
                'genre_Adventure': 0,
                'genre_Animation': 0,
                'genre_Biography': 0,
                'genre_Comedy': 0,
                'genre_Crime': 0,
                'genre_Documentary': 0,
                'genre_Drama': 0,
                'genre_Family': 0,
                'genre_Fantasy': 0,
                'genre_Film-Noir': 0,
                'genre_History': 0,
                'genre_Horror': 0,
                'genre_Music': 0,
                'genre_Musical': 0,
                'genre_Mystery': 0,
                'genre_Romance': 0,
                'genre_Sci-Fi': 0,
                'genre_Sport': 0,
                'genre_Thriller': 0,
                'genre_War': 0,
                'genre_Western': 0,
                'language_Cantonese':0,
                'language_Dutch':0,
                'language_English':0,
                'language_Finnish':0,
                'language_French':0,
                'language_German':0,
                'language_Hindi':0,
                'language_Italian':0,
                'language_Japanese':0,
                'language_Korean':0,
                'language_Mandarin':0,
                'language_Portuguese':0,
                'language_Russian':0,
                'language_Spanish':0,
                'language_Turkish':0,
                'production_company_Amazon':0,
                'production_company_BBC Films':0,
                'production_company_CJ Entertainment':0,
                'production_company_Canal+':0,
                'production_company_Constantin Film':0,
                'production_company_De Laurentiis':0,
                'production_company_Dimension Films':0,
                'production_company_Disney':0,
                'production_company_EuropaCorp':0,
                'production_company_Gaumont':0,
                'production_company_Lionsgate':0,
                'production_company_MGM':0,
                'production_company_Medusa':0,
                'production_company_Millennium Films':0,
                'production_company_Morgan Creek Entertainment':0,
                'production_company_Paramount':0,
                'production_company_RKO':0,
                'production_company_Sony':0,
                'production_company_Twentieth Century Fox':0,
                'production_company_United Artists':0,
                'production_company_Universal':0,
                'production_company_Warner':0,
                'month_published_1':0,
                'month_published_10':0,
                'month_published_11':0,
                'month_published_12':0,
                'month_published_2':0,
                'month_published_3':0,
                'month_published_4':0,
                'month_published_5':0,
                'month_published_6':0,
                'month_published_7':0,
                'month_published_8':0,
                'month_published_9':0

            })
            data_reg = multi_value_one_hot(data_reg, 'genre')
            data_reg.drop(columns=['genre'], inplace=True)
            data_reg['month_published'] = data_reg['month_published'].astype(str)
            data_reg = one_hot_encode_and_replace(data_reg, 'language')
            data_reg = one_hot_encode_and_replace(data_reg, 'production_company')
            data_reg = one_hot_encode_and_replace(data_reg, 'month_published')

            revenue= regressor.predict(data_reg)
            # messagebox.showinfo("Generated Entry", data.to_string(index=False))
            prediction = classifier.predict(data)
            if(prediction[0] == 3):
                prediction="Low"
            elif (prediction[0] == 1):
                prediction = "Medium-Low"
            elif (prediction[0] == 0):
                prediction = "Medium-High"
            elif (prediction[0] == 2):
                prediction = "High"
            messagebox.showinfo("Prediction", f"The expected earnings of the movie are on the {prediction} end. The expected revenue is {revenue[0]}")
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))

    def add_actor_field(self, parent, removable=True, index=1):
        if len(self.actor_frames) >= 10:
            return

        frame = ttk.Frame(parent)
        frame.grid(row=index, column=0, columnspan=3, sticky="ew", pady=2)
        cb = ttk.Combobox(frame, values=actor_list, state="readonly")
        cb.grid(row=0, column=0, padx=5, pady=5, sticky='W')
        cb.bind("<<ComboboxSelected>>", self.check_conditions)
        add_button = ttk.Button(frame, text="Add",
                                command=lambda: self.add_actor_field(parent, index=len(self.actor_frames) + 1))
        add_button.grid(row=0, column=1, padx=5)

        if removable:
            remove_button = ttk.Button(frame, text="Remove",
                                       command=lambda: self.remove_field(frame, self.actor_frames))
            remove_button.grid(row=0, column=2, padx=5)

        self.actor_frames.append(frame)

        if len(self.actor_frames) == 10:
            add_button.state(['disabled'])

    def add_genre_field(self, parent, removable=True, index=1):
        if len(self.genre_frames) >= 5:
            return

        frame = ttk.Frame(parent)
        frame.grid(row=index, column=0, columnspan=3, sticky="ew", pady=2)
        cb = ttk.Combobox(frame, values=genre_list, state="readonly")
        cb.grid(row=0, column=0, padx=5, pady=5, sticky='W')
        cb.bind("<<ComboboxSelected>>", self.check_conditions)
        add_button = ttk.Button(frame, text="Add",
                                command=lambda: self.add_genre_field(parent, index=len(self.genre_frames) + 301))
        add_button.grid(row=0, column=1, padx=5)

        if removable:
            remove_button = ttk.Button(frame, text="Remove",
                                       command=lambda: self.remove_field(frame, self.genre_frames))
            remove_button.grid(row=0, column=2, padx=5)

        self.genre_frames.append(frame)

        if len(self.genre_frames) == 10:
            add_button.state(['disabled'])

    def add_writer_field(self, parent, removable=True, index=201):
        if len(self.writer_frames) >= 3:
            return

        frame = ttk.Frame(parent)
        frame.grid(row=index, column=0, columnspan=3, sticky="ew", pady=2)
        cb = ttk.Combobox(frame, values=writer_list, state="readonly")
        cb.grid(row=0, column=0, padx=5, pady=5, sticky='W')
        cb.bind("<<ComboboxSelected>>", self.check_conditions)
        add_button = ttk.Button(frame, text="Add",
                                command=lambda: self.add_writer_field(parent, index=len(self.writer_frames) + 202))
        add_button.grid(row=0, column=1, padx=5)

        if removable:
            remove_button = ttk.Button(frame, text="Remove",
                                       command=lambda: self.remove_field(frame, self.writer_frames))
            remove_button.grid(row=0, column=2, padx=5)

        self.writer_frames.append(frame)

        if len(self.writer_frames) == 3:
            add_button.state(['disabled'])

    def remove_field(self, frame, list_frames):
        frame.destroy()
        list_frames.remove(frame)
        self.check_conditions()

    def check_conditions(self, event=None):
        actor_filled = any(frame.winfo_children()[0].get() for frame in self.actor_frames)
        genre_filled = any(frame.winfo_children()[0].get() for frame in self.genre_frames)
        writer_filled = any(frame.winfo_children()[0].get() for frame in self.writer_frames)
        director_filled = bool(self.director_cb.get())
        language_filled = bool(self.language_cb.get())
        company_filled = bool(self.company_cb.get())

        if actor_filled and writer_filled and director_filled and genre_filled and language_filled and company_filled:
            self.predict_button.state(['!disabled'])
        else:
            self.predict_button.state(['disabled'])

    def update_predict_button_state(self):
        self.predict_button.state(['disabled'])

    def get_awards_info(self, actors, director, writers):
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

    def create_entry(self):
        try:
            duration = int(self.duration_label.cget("text"))
            converted_budget = float(self.budget_label.cget("text").replace(",", ""))
            director = self.director_cb.get()
            genres = self.genre_cb.get().split(', ')
            language = self.language_cb.get()
            production_company = self.company_cb.get()
            month_published = str(months.index(self.month_cb.get()) + 1)

            actors = [frame.winfo_children()[0].get() for frame in self.actor_frames]
            genre = [frame.winfo_children()[0].get() for frame in self.genre_frames]
            writers = [frame.winfo_children()[0].get() for frame in self.writer_frames]

            awards_info = self.get_awards_info(actors, director, writers)

            genre_encoding = self.get_genre_encoding(genres)

            entry = {
                'duration': duration,
                'converted_budget': converted_budget,
                'dir_oscar_nomination': awards_info['dir_oscar_nomination'],
                'writer_oscar_nomination': awards_info['writer_oscar_nomination'],
                'cast_globe_nomination': awards_info['cast_globe_nomination'],
                'BAFTA_act_nom': awards_info['BAFTA_act_nom'],
                'BAFTA_dir_nom': awards_info['BAFTA_dir_nom'],
                'BAFTA_writer_nom': awards_info['BAFTA_writer_nom'],
                'dir_emmy_nom': awards_info['dir_emmy_nom'],
                'writer_emmy_nom': awards_info['writer_emmy_nom'],
                'act_emmy_nom': awards_info['act_emmy_nom'],
                'actors_films_before': awards_info['actors_films_before'],
                'director_films_before': awards_info['director_films_before'],
                'writers_films_before': awards_info['writers_films_before'],
                'language': language,
                'production_company': production_company,
                'month_published': month_published,
                'revenue_cluster': ''  # Placeholder, update based on your logic
            }
            entry.update(genre_encoding)

            # Convert to DataFrame for better visualization
            entry_df = pd.DataFrame([entry])

            # Show the entry in a message box
            # messagebox.showinfo("Generated Entry", entry_df.to_string(index=False))

        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))

    def get_genre_encoding(self, genres):
        all_genres = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
                      'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance',
                      'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
        genre_encoding = {f'genre_{genre}': 0 for genre in all_genres}
        for genre in genres:
            genre_encoding[f'genre_{genre}'] = 1
        return genre_encoding

root = tk.Tk()
app = DynamicFieldsApp(root)
root.mainloop()
