# database_manager.py
import sqlite3
from pathlib import Path
import pandas as pd


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = None

        try:
            self.connection = sqlite3.connect(self.db_path)
            print(f"Pomyślnie połączono z bazą danych: {self.db_path}")
            self._create_table_if_not_exists()
        except sqlite3.Error as e:
            print(f"Błąd krytyczny podczas łączenia z bazą danych: {e}")
            raise

    def _create_table_if_not_exists(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ref_path TEXT NOT NULL,
                deg_path TEXT NOT NULL UNIQUE,
                bitrate TEXT,
                noise_level REAL,
                filter_cutoff REAL,
                mos_lqo REAL,
                odg REAL,
                cnn_1d_score REAL,
                efficientnet_score REAL,
                inception_score REAL,
                vgg19_score REAL,
                status TEXT
            )
        """)
        self.connection.commit()
        print("Tabela 'audio_pairs' jest gotowa.")

    def add_pair(self, pair_data):
        sql = '''INSERT INTO audio_pairs(ref_path, deg_path, bitrate, noise_level, filter_cutoff, status)
                 VALUES(:ref_path, :deg_path, :bitrate, :noise_level, :filter_cutoff, :status)'''
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, pair_data)
            self.connection.commit()
            print(f"Dodano nową parę do bazy danych: {pair_data['deg_path']}")
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"Błąd: Próba dodania zduplikowanego pliku: {pair_data['deg_path']}")
            return None
        except sqlite3.Error as e:
            print(f"Błąd podczas dodawania pary do bazy danych: {e}")
            return None

    def get_all_pairs_as_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_sql_query("SELECT * FROM audio_pairs", self.connection)
            print(f"Pomyślnie wczytano {len(df)} wierszy z bazy danych do DataFrame.")
            return df
        except Exception as e:
            print(f"Błąd podczas wczytywania danych do DataFrame: {e}")
            return pd.DataFrame()

    def update_subjective_score(self, deg_path, score):
        sql = "UPDATE audio_pairs SET subjective_score = ? WHERE deg_path = ?"
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, (score, deg_path))
            self.connection.commit()
            print(f"Zapisano ocenę {score} dla pliku {deg_path}")
            return True
        except sqlite3.Error as e:
            print(f"Błąd podczas zapisu oceny subiektywnej: {e}")
            return False

    def update_analysis_results(self, deg_path, results: dict):
        # lista dozwolonych kolumn do modyfikacji przy update wynikow
        allowed_columns = ['mos_lqo', 'odg', 'cnn_1d_score', 'efficientnet_score',
                           'inception_score', 'vgg19_score', 'status']

        # dynamicznie zapytanie tylko dla kluczy, które są w allowed_columns
        set_clauses = [f"{col} = :{col}" for col in results.keys() if col in allowed_columns]

        if not set_clauses:
            print("Brak wyników do zaktualizowania (lub brak pasujących kolumn).")
            return

        sql = f"UPDATE audio_pairs SET {', '.join(set_clauses)} WHERE deg_path = :deg_path"

        params = results.copy()
        params['deg_path'] = deg_path

        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            self.connection.commit()
            print(f"Zaktualizowano wyniki dla pliku: {deg_path}")
        except sqlite3.Error as e:
            print(f"Błąd podczas aktualizacji wyników: {e}")

    def clear_analysis_results(self):
        sql = ''' UPDATE audio_pairs
                  SET mos_lqo = NULL,
                      odg = NULL,
                      cnn_1d_score = NULL,
                      efficientnet_score = NULL,
                      inception_score = NULL,
                      vgg19_score = NULL,
                      status = 'Gotowy do analizy'
              '''
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            print(f"Wyczyszczono wyniki analizy dla {cursor.rowcount} wierszy.")
            return True
        except sqlite3.Error as e:
            print(f"Błąd podczas czyszczenia wyników: {e}")
            return False

    def close(self):
        if self.connection:
            self.connection.close()
            print("Zamknięto połączenie z bazą danych.")