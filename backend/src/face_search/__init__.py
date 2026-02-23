"""
face_search — Offline, GPU-accelerated face recognition and organization.
"""

__version__ = "0.1.0"


def sort_images_by_person(*args, **kwargs):
    """Sort images in a folder by detected person identity.

    See face_search._operations.sort_images_by_person for full docs.
    """
    from ._operations import sort_images_by_person as _sort

    return _sort(*args, **kwargs)


def find_person_folder(*args, **kwargs):
    """Find which person folder a query image belongs to.

    See face_search._operations.find_person_folder for full docs.
    """
    from ._operations import find_person_folder as _find

    return _find(*args, **kwargs)
