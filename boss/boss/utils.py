from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from flask import json

# from boss.models import StepEstimate


def serialize_task_to_string(task):
    return json.dumps(serialize_task(task))


def serialize_task(task):
    """
    Recursively converts ObjectId, datetime, and Anthropic TextBlock instances in a MongoDB document to strings,
    ensuring datetimes are in ISO 8601 format with 'Z' at the end.
    """
    # if isinstance(task, StepEstimate):
    #     return task.model_dump()  # Convert the object to a dictionary
    if isinstance(task, dict):
        return {k: serialize_task(v) for k, v in task.items()}
    elif isinstance(task, list):
        return [serialize_task(element) for element in task]
    elif isinstance(task, ObjectId):
        return str(task)
    elif isinstance(task, datetime):
        if task.tzinfo is None:
            task = task.replace(tzinfo=timezone.utc)
        else:
            task = task.astimezone(timezone.utc)
        iso_string = task.isoformat()
        if iso_string.endswith("+00:00"):
            iso_string = iso_string[:-6] + "Z"
        elif iso_string.endswith("+00"):
            iso_string = iso_string[:-3] + "Z"
        else:
            iso_string += "Z"
        return iso_string
    # Handle Anthropic TextBlock objects
    elif hasattr(task, "text") and hasattr(task, "type"):
        return task.text
    else:
        return task

    # Works as expected


def ensure_timezone_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware, defaulting to UTC"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_iso_timestamp(input_dt: Optional[datetime] = None) -> str:
    """Get a timezone-aware ISO 8601 timestamp with 'Z' at the end"""
    dt = input_dt or datetime.now(timezone.utc)
    iso_string = dt.isoformat()
    if iso_string.endswith("+00:00"):
        iso_string = iso_string[:-6] + "Z"
    return iso_string


def get_object_id(id_value) -> Optional[ObjectId]:
    """Safely convert string to ObjectId"""
    if isinstance(id_value, ObjectId):
        return id_value
    try:
        return ObjectId(id_value)
    except Exception:
        return None
