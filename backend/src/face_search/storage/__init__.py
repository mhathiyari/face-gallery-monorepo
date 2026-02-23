"""Storage layer for face search system.

This module provides storage solutions for face embeddings and metadata:
- FAISS index for fast similarity search
- Metadata database for face information
- Collection management for organizing faces

Usage:
    from face_search.storage import Collection

    # Create or load a collection
    collection = Collection(
        collection_id='my_collection',
        collections_root='./collections',
        embedding_dim=512
    )

    # Add a face
    face_id = collection.add_face(
        image_path='/path/to/image.jpg',
        bbox={'x1': 10, 'y1': 20, 'x2': 100, 'y2': 150},
        embedding=embedding_vector,
        confidence=0.95
    )

    # Search for similar faces
    results = collection.search(query_embedding, k=10)
"""

from .faiss_index import FAISSIndex, IndexType
from .metadata_db import MetadataStore, Face
from .collection_lock import CollectionLock, collection_lock, LockTimeout
from .collection_manager import Collection, SyncError

__all__ = [
    'FAISSIndex',
    'IndexType',
    'MetadataStore',
    'Face',
    'CollectionLock',
    'collection_lock',
    'LockTimeout',
    'Collection',
    'SyncError',
]
