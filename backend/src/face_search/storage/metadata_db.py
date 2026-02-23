"""Metadata database for storing face information.

This module provides a SQLite database interface for storing face metadata,
including image paths, bounding boxes, embeddings, and other face attributes.
The database is synchronized with the FAISS index via embedding_index field.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import logging
import pickle

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    LargeBinary,
    DateTime,
    Index as DBIndex,
    func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import numpy as np

logger = logging.getLogger(__name__)

Base = declarative_base()


class Face(Base):
    """Face metadata table.

    Stores information about detected faces, including their location in images,
    embeddings, and synchronization with the FAISS index.
    """
    __tablename__ = 'faces'

    # Primary key
    face_id = Column(Integer, primary_key=True, autoincrement=True)

    # Collection ID (for multi-collection support)
    collection_id = Column(String(255), nullable=False, index=True)

    # Image information
    image_path = Column(String(1024), nullable=False, index=True)

    # Bounding box (stored as JSON)
    bbox = Column(String(256), nullable=False)  # JSON: {x1, y1, x2, y2}

    # Embedding stored as BLOB
    embedding = Column(LargeBinary, nullable=False)

    # Index in FAISS (for synchronization)
    # Note: unique constraint is per collection, not global
    embedding_index = Column(Integer, nullable=False, index=True)

    # Detection confidence
    confidence = Column(Float, nullable=False)

    # Face pose angles (for quality assessment)
    yaw = Column(Float, nullable=True)    # Left-right rotation (-90 to +90)
    pitch = Column(Float, nullable=True)  # Up-down tilt
    roll = Column(Float, nullable=True)   # Head tilt angle

    # Cluster ID (for grouping faces by identity)
    cluster_id = Column(Integer, nullable=True, index=True)

    # Soft delete flag
    deleted = Column(Boolean, default=False, nullable=False, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Create composite indexes
    # Note: unique constraint on embedding_index is created as partial index
    # (only for non-deleted faces) in MetadataStore.__init__
    __table_args__ = (
        DBIndex('idx_collection_deleted', 'collection_id', 'deleted'),
        DBIndex('idx_collection_image', 'collection_id', 'image_path'),
        DBIndex('idx_collection_embedding', 'collection_id', 'embedding_index'),
    )

    def to_dict(self) -> dict:
        """Convert face record to dictionary.

        Returns:
            Dictionary representation of the face
        """
        return {
            'face_id': self.face_id,
            'collection_id': self.collection_id,
            'image_path': self.image_path,
            'bbox': json.loads(self.bbox) if self.bbox else None,
            'embedding_index': self.embedding_index,
            'confidence': self.confidence,
            'deleted': self.deleted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    def get_embedding(self) -> np.ndarray:
        """Deserialize embedding from BLOB.

        Returns:
            Embedding as numpy array
        """
        return pickle.loads(self.embedding)

    def set_embedding(self, embedding: np.ndarray):
        """Serialize embedding to BLOB.

        Args:
            embedding: Numpy array to store
        """
        self.embedding = pickle.dumps(embedding)


class MetadataStore:
    """Database store for face metadata.

    Manages face records in SQLite database with support for:
    - CRUD operations
    - Soft deletes
    - FAISS index synchronization
    - Transaction management
    """

    def __init__(self, db_path: str, collection_id: str = "default"):
        """Initialize metadata store.

        Args:
            db_path: Path to SQLite database file
            collection_id: Collection identifier
        """
        self.db_path = Path(db_path)
        self.collection_id = collection_id

        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine and session
        self.engine = create_engine(
            f'sqlite:///{self.db_path}',
            echo=False,
            connect_args={'check_same_thread': False}  # For SQLite
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create partial unique index for (collection_id, embedding_index)
        # Only applies to non-deleted faces to avoid conflicts during rebuild
        from sqlalchemy import text as sql_text
        with self.engine.connect() as conn:
            # Check if index exists
            result = conn.execute(
                sql_text(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_unique_active_embedding'"
                )
            )
            if not result.fetchone():
                conn.execute(
                    sql_text(
                        "CREATE UNIQUE INDEX idx_unique_active_embedding "
                        "ON faces(collection_id, embedding_index) "
                        "WHERE deleted = 0"
                    )
                )
                conn.commit()

        logger.info(f"MetadataStore initialized: {self.db_path}")

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations.

        Usage:
            with store.session_scope() as session:
                session.add(face)
                # Commit happens automatically on success
                # Rollback happens automatically on exception
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def add_face(
        self,
        image_path: str,
        bbox: Dict[str, float],
        embedding: np.ndarray,
        embedding_index: int,
        confidence: float,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None
    ) -> int:
        """Add a face record to the database.

        Args:
            image_path: Path to the source image
            bbox: Bounding box dictionary {x1, y1, x2, y2}
            embedding: Face embedding as numpy array
            embedding_index: Index in FAISS
            confidence: Detection confidence score
            yaw: Face yaw angle (left-right rotation)
            pitch: Face pitch angle (up-down tilt)
            roll: Face roll angle (head tilt)

        Returns:
            Face ID of the created record

        Raises:
            SQLAlchemyError: If database operation fails
        """
        with self.session_scope() as session:
            face = Face(
                collection_id=self.collection_id,
                image_path=str(image_path),
                bbox=json.dumps(bbox),
                embedding_index=embedding_index,
                confidence=confidence,
                yaw=yaw,
                pitch=pitch,
                roll=roll
            )
            face.set_embedding(embedding)

            session.add(face)
            session.flush()  # Get the ID before commit
            face_id = face.face_id

        logger.debug(f"Added face {face_id} from {image_path}")
        return face_id

    def get_face_by_id(self, face_id: int) -> Optional[Face]:
        """Get face record by ID.

        Args:
            face_id: Face ID

        Returns:
            Face record or None if not found
        """
        with self.session_scope() as session:
            face = session.query(Face).filter(
                Face.face_id == face_id,
                Face.collection_id == self.collection_id
            ).first()
            if face:
                # Detach from session
                session.expunge(face)
            return face

    def get_face_by_index(self, embedding_index: int) -> Optional[Face]:
        """Get face record by FAISS embedding index.

        Args:
            embedding_index: Index in FAISS

        Returns:
            Face record or None if not found
        """
        with self.session_scope() as session:
            face = session.query(Face).filter(
                Face.embedding_index == embedding_index,
                Face.collection_id == self.collection_id
            ).first()
            if face:
                session.expunge(face)
            return face

    def get_faces_by_image(self, image_path: str) -> List[Face]:
        """Get all faces from a specific image.

        Args:
            image_path: Path to the image

        Returns:
            List of Face records
        """
        with self.session_scope() as session:
            faces = session.query(Face).filter(
                Face.image_path == str(image_path),
                Face.collection_id == self.collection_id,
                Face.deleted == False
            ).all()
            # Detach from session
            for face in faces:
                session.expunge(face)
            return faces

    def mark_deleted(self, face_id: int) -> bool:
        """Mark a face as deleted (soft delete).

        Args:
            face_id: Face ID to mark as deleted

        Returns:
            True if successful, False if face not found
        """
        with self.session_scope() as session:
            face = session.query(Face).filter(
                Face.face_id == face_id,
                Face.collection_id == self.collection_id
            ).first()

            if face:
                face.deleted = True
                face.updated_at = datetime.utcnow()
                logger.debug(f"Marked face {face_id} as deleted")
                return True
            return False

    def delete_face(self, face_id: int) -> bool:
        """Permanently delete a face record.

        Args:
            face_id: Face ID to delete

        Returns:
            True if successful, False if face not found
        """
        with self.session_scope() as session:
            face = session.query(Face).filter(
                Face.face_id == face_id,
                Face.collection_id == self.collection_id
            ).first()

            if face:
                session.delete(face)
                logger.debug(f"Permanently deleted face {face_id}")
                return True
            return False

    def update_cluster_assignments(self, face_cluster_map: dict) -> int:
        """Update cluster_id for multiple faces in batch.

        Args:
            face_cluster_map: Dictionary mapping face_id to cluster_id

        Returns:
            Number of faces updated
        """
        with self.session_scope() as session:
            updated = 0
            for face_id, cluster_id in face_cluster_map.items():
                face = session.query(Face).filter(
                    Face.face_id == face_id,
                    Face.collection_id == self.collection_id
                ).first()

                if face:
                    face.cluster_id = cluster_id
                    updated += 1

            logger.info(f"Updated cluster_id for {updated} faces")
            return updated

    def count_active_faces(self) -> int:
        """Count active (non-deleted) faces in the collection.

        Returns:
            Number of active faces
        """
        with self.session_scope() as session:
            count = session.query(func.count(Face.face_id)).filter(
                Face.collection_id == self.collection_id,
                Face.deleted == False
            ).scalar()
            return count or 0

    def count_total_faces(self) -> int:
        """Count total faces (including deleted) in the collection.

        Returns:
            Total number of faces
        """
        with self.session_scope() as session:
            count = session.query(func.count(Face.face_id)).filter(
                Face.collection_id == self.collection_id
            ).scalar()
            return count or 0

    def get_all_faces(self, include_deleted: bool = False) -> List[Face]:
        """Get all faces in the collection.

        Args:
            include_deleted: Whether to include deleted faces

        Returns:
            List of Face records
        """
        with self.session_scope() as session:
            query = session.query(Face).filter(
                Face.collection_id == self.collection_id
            )

            if not include_deleted:
                query = query.filter(Face.deleted == False)

            faces = query.all()
            # Detach from session
            for face in faces:
                session.expunge(face)
            return faces

    def get_deleted_faces(self) -> List[Face]:
        """Get all deleted faces in the collection.

        Returns:
            List of deleted Face records
        """
        with self.session_scope() as session:
            faces = session.query(Face).filter(
                Face.collection_id == self.collection_id,
                Face.deleted == True
            ).all()
            # Detach from session
            for face in faces:
                session.expunge(face)
            return faces

    def get_embedding_by_index(self, embedding_index: int) -> Optional[np.ndarray]:
        """Get embedding by FAISS index.

        Args:
            embedding_index: Index in FAISS

        Returns:
            Embedding as numpy array or None if not found
        """
        face = self.get_face_by_index(embedding_index)
        if face:
            return face.get_embedding()
        return None

    def rebuild_index_mapping(self) -> Dict[int, int]:
        """Get mapping of old to new indices for index rebuild.

        This is used when rebuilding the FAISS index to remove deleted faces.

        Returns:
            Dictionary mapping old_index -> new_index for active faces that change
            (only includes entries where old_index != new_index)
        """
        with self.session_scope() as session:
            active_faces = session.query(Face).filter(
                Face.collection_id == self.collection_id,
                Face.deleted == False
            ).order_by(Face.embedding_index).all()

            # Create mapping: old_index -> new_index
            # Only include indices that actually change
            mapping = {}
            for new_idx, face in enumerate(active_faces):
                if face.embedding_index != new_idx:
                    mapping[face.embedding_index] = new_idx

            return mapping

    def update_indices_after_rebuild(self, index_mapping: Dict[int, int]):
        """Update embedding indices after FAISS index rebuild.

        Uses a two-phase raw SQL update to avoid unique constraint violations:
        1. Move all to negative temporary values
        2. Move from temporary to final values

        Args:
            index_mapping: Dictionary mapping old_index -> new_index
        """
        if not index_mapping:
            return

        from sqlalchemy import text

        with self.session_scope() as session:
            temp_offset = -1000000

            # Phase 1: Move all to negative temporary values
            case_whens_temp = []
            old_indices = []
            for i, old_idx in enumerate(index_mapping.keys()):
                temp_idx = temp_offset + i
                case_whens_temp.append(f"WHEN embedding_index = {old_idx} THEN {temp_idx}")
                old_indices.append(str(old_idx))

            case_statement_temp = " ".join(case_whens_temp)
            in_clause = ", ".join(old_indices)

            sql_temp = f"""
                UPDATE faces
                SET embedding_index = CASE {case_statement_temp} END
                WHERE collection_id = :collection_id
                AND embedding_index IN ({in_clause})
            """

            session.execute(
                text(sql_temp),
                {'collection_id': self.collection_id}
            )

            # Flush phase 1
            session.flush()

            # Phase 2: Move from temporary to final values
            case_whens_final = []
            temp_indices = []
            for i, (old_idx, new_idx) in enumerate(index_mapping.items()):
                temp_idx = temp_offset + i
                case_whens_final.append(f"WHEN embedding_index = {temp_idx} THEN {new_idx}")
                temp_indices.append(str(temp_idx))

            case_statement_final = " ".join(case_whens_final)
            in_clause_temp = ", ".join(temp_indices)

            sql_final = f"""
                UPDATE faces
                SET embedding_index = CASE {case_statement_final} END,
                    updated_at = :updated_at
                WHERE collection_id = :collection_id
                AND embedding_index IN ({in_clause_temp})
            """

            session.execute(
                text(sql_final),
                {
                    'collection_id': self.collection_id,
                    'updated_at': datetime.utcnow()
                }
            )

        logger.info(f"Updated {len(index_mapping)} face indices after rebuild")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'collection_id': self.collection_id,
            'total_faces': self.count_total_faces(),
            'active_faces': self.count_active_faces(),
            'deleted_faces': self.count_total_faces() - self.count_active_faces(),
            'db_path': str(self.db_path),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MetadataStore(collection='{self.collection_id}', "
            f"faces={self.count_active_faces()}/{self.count_total_faces()})"
        )
