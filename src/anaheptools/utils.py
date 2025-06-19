from collections.abc import Collection


def is_collection(obj: object) -> bool:
    """Check if the object is a collection (list, tuple, set, etc.) but not a string."""
    return isinstance(obj, Collection) and not isinstance(obj, str | bytes)
