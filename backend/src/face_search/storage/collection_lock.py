"""File-based locking for collection access.

This module provides file-based locking to prevent concurrent modifications
to collections, ensuring data consistency between FAISS index and metadata database.
"""

import os
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class LockTimeout(Exception):
    """Raised when lock acquisition times out."""
    pass


class CollectionLock:
    """File-based lock for collection access.

    Uses file-based locking to prevent concurrent modifications to a collection.
    This ensures that the FAISS index and metadata database stay synchronized.

    Attributes:
        lock_path: Path to the lock file
        timeout: Maximum time to wait for lock acquisition (seconds)
        poll_interval: Time between lock acquisition attempts (seconds)
    """

    def __init__(
        self,
        lock_path: Path,
        timeout: float = 30.0,
        poll_interval: float = 0.1
    ):
        """Initialize collection lock.

        Args:
            lock_path: Path where lock file will be created
            timeout: Maximum time to wait for lock (seconds)
            poll_interval: Time between lock attempts (seconds)
        """
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._lock_fd: Optional[int] = None

    def acquire(self) -> bool:
        """Acquire the lock.

        Returns:
            True if lock acquired successfully

        Raises:
            LockTimeout: If lock cannot be acquired within timeout
        """
        start_time = time.time()

        # Create lock directory if needed
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # Try to create lock file exclusively
                # O_CREAT | O_EXCL ensures atomic creation
                self._lock_fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )

                # Write PID to lock file for debugging
                os.write(self._lock_fd, str(os.getpid()).encode())
                logger.debug(f"Lock acquired: {self.lock_path}")
                return True

            except FileExistsError:
                # Lock file already exists, check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    raise LockTimeout(
                        f"Could not acquire lock {self.lock_path} "
                        f"within {self.timeout} seconds"
                    )

                # Wait before retrying
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error acquiring lock: {e}")
                raise

    def release(self):
        """Release the lock."""
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
                self._lock_fd = None
            except Exception as e:
                logger.warning(f"Error closing lock file descriptor: {e}")

        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
                logger.debug(f"Lock released: {self.lock_path}")
        except Exception as e:
            logger.warning(f"Error removing lock file: {e}")

    def is_locked(self) -> bool:
        """Check if lock file exists.

        Returns:
            True if lock file exists (may be stale)
        """
        return self.lock_path.exists()

    def force_unlock(self):
        """Force remove the lock file.

        Warning: Only use this if you're sure the lock is stale!
        """
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
                logger.warning(f"Lock forcefully removed: {self.lock_path}")
        except Exception as e:
            logger.error(f"Error force unlocking: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False  # Don't suppress exceptions

    def __repr__(self) -> str:
        """String representation."""
        status = "locked" if self.is_locked() else "unlocked"
        return f"CollectionLock({self.lock_path}, status={status})"


@contextmanager
def collection_lock(
    collection_path: Path,
    timeout: float = 30.0
):
    """Context manager for collection locking.

    Usage:
        with collection_lock(collection_path):
            # Safe to modify collection
            pass

    Args:
        collection_path: Path to collection directory
        timeout: Maximum time to wait for lock

    Yields:
        CollectionLock instance

    Raises:
        LockTimeout: If lock cannot be acquired
    """
    lock_path = collection_path / ".lock"
    lock = CollectionLock(lock_path, timeout=timeout)

    try:
        lock.acquire()
        yield lock
    finally:
        lock.release()
