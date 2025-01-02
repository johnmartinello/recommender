from typing import Dict, Optional
import psycopg2
import logging
from database.vector_db import VectorDB

conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="123",
        host="localhost",
        port="5432"
        )

def search_movies(
    query: str,
    metadata: Optional[Dict] = None
    ):
    
    logger = logging.getLogger(__name__)
    vector_db = VectorDB()

    if metadata and "time_range" in metadata:
        try:
            start_year, end_year = map(int, metadata["time_range"].split('-'))
            metadata["time_range"] = (start_year, end_year)
        except ValueError as e:
            logger.error(f"Invalid time_range format: {metadata['time_range']}")
            raise e

    results = vector_db.search(
        conn=conn,
        query_text=query,
        limit=5,
        metadata=metadata,
    )
        
    return results

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    query = "A kid accidentally accesses the Pentagon servers"
    metadata = {
        "genres": "Family, Comedy",
        "time_range": "1960-2020"
    }
    
    
    results = search_movies(query,metadata)
    
    for result in results:
        print(f"{result[1]}, {result[4]}")


if __name__ == "__main__":
    main()