from datetime import datetime, timezone
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
import logging
import json
import ast


class MovieVectorDB:
    def __init__(self, csv_path):
        """Initialize the MovieVectorDB with path to CSV file."""
        self.csv_path = csv_path
        self.vec_store = VectorStore()
        self.logger = logging.getLogger(__name__)
        
    def load_and_clean_data(self):
        """Load and clean the movie dataset."""
        try:
            df = pd.read_csv(self.csv_path, sep=",")
            
            df = df.dropna(subset=['overview'])
            df = df[df['overview'].apply(lambda x: isinstance(x, str))]

            df = df.dropna(subset=['genres'])
            df = df[df['genres'].apply(lambda x: isinstance(x, str))]
            df = df[df['genres'].str.len() > 0]

            df = df.dropna(subset=['keywords'])
            df = df[df['keywords'].apply(lambda x: isinstance(x, str))]
            df = df[df['keywords'].str.len() > 0]

            df = df[df['popularity'] > 1]

            df = df.drop_duplicates(subset=['id'])

            df = df[(df['runtime'] > 0) & (df['runtime'] < 300)]  

            df = df[
            (pd.to_datetime(df['release_date']).dt.year >= 1900) & 
            (pd.to_datetime(df['release_date']).dt.year <= 2024)]

            df = df[df['vote_count'] > 10]

            df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)]

            self.logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

            return df
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_record(self, row):
        """Transform a row into the format expected by TimescaleVector."""
        try:

            
            release_date = datetime.now(timezone.utc) 
            if pd.notna(row['release_date']):
                try:
                    release_date = pd.to_datetime(row['release_date']).tz_localize('UTC')
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid date format for movie: {row.get('title', 'Unknown')}, using current time")
            
            
            metadata = {
                "title": str(row.get('title', '')).strip(), 
                "movie_id": str(row.get('id', '')).strip(),
                "genres": str(row.get('genres', '')),
                "popularity": row.get('popularity', 0.0),
                "poster_path": str(row.get('poster_path', '')).strip(),
                "production_companies": str(row.get('production_companies', '')),
                "production_countries": str(row.get('production_countries', '')),
                "original_language": str(row.get('original_language', '')).strip(),
                "keywords": str(row.get('keywords', '')),
                
                "created_at": release_date.isoformat()
            }

            embedding = self.vec_store.get_embedding_local(row['overview'])
            return {
                "id": str(uuid_from_time(release_date)),
                "metadata": metadata, 
                "contents": str(row['overview']).strip(),  
                "embedding": embedding,
            }
        except Exception as e:
            self.logger.error(f"Error preparing record: {str(e)}")
            raise
            
    def process_batch(self, df, batch_size=1000):
        """Process records in batches to manage memory usage."""
        records = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batch_records = []
            for _, row in batch.iterrows():
                try:
                    record = self.prepare_record(row)
                    batch_records.append(record)
                except Exception as e:
                    self.logger.error(f"Error processing row: {str(e)}")
                    continue
            records.extend(batch_records)
            self.logger.info(f"Processed batch {i//batch_size + 1} ({(len(df)//1000) -(i//batch_size + 1) } left)")
        return pd.DataFrame(records)
    

            
            
    def setup_database(self):
        """Set up the vector database with movie data."""
        try:
            
            records_df = self.load_and_clean_data()
            records_df = self.process_batch(records_df)
            
            self.vec_store.create_tables()
            self.vec_store.create_index()
            
            self.vec_store.upsert(records_df)
            self.logger.info("Successfully inserted data into vector store")
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    movie_db = MovieVectorDB("D:/recommender/data/tmdb.csv")
    movie_db.setup_database()

if __name__ == "__main__":
    main()