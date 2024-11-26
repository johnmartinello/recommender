# similarity_search.py
import psycopg2
import logging
from datetime import datetime
from database.vector_store import VectorStore
from typing import List, Dict

def search_movies(queries: List[Dict[str, str]]) -> None:
    """
    Search for movies based on a list of query parameters.
    Each query should contain: title, overview, genres, keywords
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Instantiate VectorStore
    vector_store = VectorStore()
    
    # Default time range
    time_range = (
        datetime(1970, 1, 1),
        datetime(2024, 12, 31)
    )
    
    # Process each query
    for query in queries:
        logger.info(f"\nSearching for movies similar to: {query['title']}")
        
        # Prepare metadata filter
        metadata_filter = {
            "genres": query.get('genres', ''),
            "keywords": query.get('keywords', '')
        }
        
        # Perform the search

        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="123",
            host="localhost",
            port="5432"
        )

        results = vector_store.search(
            conn = conn,
            query_text=query['overview'],
            limit=5,
            metadata_filter=metadata_filter,
            time_range=time_range,
        )
        
        # Display results
        logger.info(f"Results for query: {query['title']}")
        for result in results:
            print(f"Title: {result[1]['title']}")
            print(f"Similarity: {result[3]:.3f}\n")
        print()
        print("-------------------------------")

def main():
    # Example queries
    queries = [
        {
            "title": "",
            "overview": "A kid accidentaly access the pentagon servers",
            "genres": "",
            "keywords": "kid, pentagon, hacker"
        },
   
    ] 
    
    search_movies(queries)

if __name__ == "__main__":
    main()