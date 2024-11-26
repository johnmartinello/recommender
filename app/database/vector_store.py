import logging
import time
from typing import Any, List, Optional, Tuple, Union, Dict
from datetime import datetime

import pandas as pd
from config.settings import get_settings
from timescale_vector import client
from sentence_transformers import SentenceTransformer
import torch


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings, local model and Timescale Vector client."""
        self.settings = get_settings()
        self.embedding_model = self.settings.local.path_model
        self.vector_settings = self.settings.vector_store
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval,
        )


    def get_embedding_local(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using local model.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        if not hasattr(self, 'model'):
            self.model = SentenceTransformer(self.settings.local.path_model)
            self.model.to(self.settings.local.device)

        # Preprocess text
        
        text = text.replace("\n", " ")
        
        start_time = time.time()
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode(
                [text],
                convert_to_tensor=True,
                device=self.settings.local.device,
                batch_size=32,
                show_progress_bar=False
            )
            # Convert to list and move to CPU if needed
            embedding = embedding[0].cpu().numpy().tolist()
        
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        
        return embedding

    def get_embedding_openAI(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tablesin the database"""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to spseed up similarity search"""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )

    def search(
        self,
        conn,
        query_text: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """Query the vector database for similar embeddings based on input text."""
    
        query_embedding = self.get_embedding_local(query_text)
        query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'

        sql_query = """
        SELECT
            id,
            metadata,
            contents,
            1 - (embedding <=> %s::vector) AS similarity
        FROM
            embeddings
        WHERE
            TRUE
        """
        params = [query_vector_str]

        if metadata_filter:
            for key, value in metadata_filter.items():
                if key == 'original_language':
                    sql_query += " AND metadata->>'original_language' = %s"
                    params.append(value)
                elif key in ('genres', 'keywords'):
                    # Split terms and create OR conditions for each
                    terms = [term.strip() for term in value.split(',')]
                    if terms:
                        conditions = []
                        for term in terms:
                            conditions.append(f"metadata->>'{key}' ILIKE %s")
                            params.append(f"%{term}%")
                        sql_query += f" AND ({' OR '.join(conditions)})"

        # Add time range filter
        if time_range:
            start_date, end_date = time_range
            sql_query += " AND (metadata->>'created_at')::timestamp BETWEEN %s AND %s"
            params.extend([start_date.isoformat(), end_date.isoformat()])

        # Order by similarity and limit the results
        sql_query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, limit])
        
        with conn.cursor() as cur:
            cur.execute(sql_query, params)
            results = cur.fetchall()

        return results

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )
