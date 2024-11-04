from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId

# Import the functions from utils.py
from utils import ensure_timezone_aware, get_object_id, serialize_task


# Mock class to simulate Anthropic TextBlock
class MockTextBlock:
    def __init__(self, text, type):
        self.text = text
        self.type = type


# Tests for serialize_task
@pytest.mark.parametrize(
    "input_value, expected",
    [
        # Test ObjectId serialization
        (ObjectId("507f1f77bcf86cd799439011"), "507f1f77bcf86cd799439011"),
        # Test datetime serialization (naive datetime)
        (datetime(2023, 10, 1, 12, 30, 45), "2023-10-01T12:30:45Z"),
        # Test datetime serialization (aware datetime in UTC)
        (
            datetime(2023, 10, 1, 12, 30, 45, tzinfo=timezone.utc),
            "2023-10-01T12:30:45Z",
        ),
        # Test datetime serialization (aware datetime in another timezone)
        (
            datetime(2023, 10, 1, 7, 30, 45, tzinfo=timezone(timedelta(hours=-5))),
            "2023-10-01T12:30:45Z",
        ),
        # Test Anthropic TextBlock serialization
        (MockTextBlock(text="Sample text", type="text"), "Sample text"),
        # Test other types (e.g., string)
        ("Just a string", "Just a string"),
        # Test list containing various types
        (
            [
                ObjectId("507f1f77bcf86cd799439011"),
                datetime(2023, 10, 1, 12, 30, 45),
                MockTextBlock(text="List text", type="text"),
                "Another string",
            ],
            [
                "507f1f77bcf86cd799439011",
                "2023-10-01T12:30:45Z",
                "List text",
                "Another string",
            ],
        ),
        # Test nested dictionaries and lists
        (
            {
                "id": ObjectId("507f1f77bcf86cd799439011"),
                "created_at": datetime(2023, 10, 1, 12, 30, 45),
                "blocks": [
                    MockTextBlock(text="Block 1", type="text"),
                    MockTextBlock(text="Block 2", type="text"),
                ],
                "details": {
                    "updated_at": datetime(
                        2023, 10, 2, 15, 45, 30, tzinfo=timezone(timedelta(hours=2))
                    ),
                    "tags": ["urgent", "review"],
                },
            },
            {
                "id": "507f1f77bcf86cd799439011",
                "created_at": "2023-10-01T12:30:45Z",
                "blocks": ["Block 1", "Block 2"],
                "details": {
                    "updated_at": "2023-10-02T13:45:30Z",
                    "tags": ["urgent", "review"],
                },
            },
        ),
    ],
)
def test_serialize_task(input_value, expected):
    assert serialize_task(input_value) == expected


# Tests for ensure_timezone_aware
@pytest.mark.parametrize(
    "input_dt, expected",
    [
        # Test with None
        (None, None),
        # Test with naive datetime
        (
            datetime(2023, 10, 1, 12, 30, 45),
            datetime(2023, 10, 1, 12, 30, 45, tzinfo=timezone.utc),
        ),
        # Test with aware datetime in UTC
        (
            datetime(2023, 10, 1, 12, 30, 45, tzinfo=timezone.utc),
            datetime(2023, 10, 1, 12, 30, 45, tzinfo=timezone.utc),
        ),
        # Test with aware datetime in another timezone
        (
            datetime(2023, 10, 1, 7, 30, 45, tzinfo=timezone(timedelta(hours=-5))),
            datetime(2023, 10, 1, 12, 30, 45, tzinfo=timezone.utc),
        ),
    ],
)
def test_ensure_timezone_aware(input_dt, expected):
    result = ensure_timezone_aware(input_dt)
    if result is not None:
        assert result == expected
    else:
        assert result is None


# Tests for get_object_id
@pytest.mark.parametrize(
    "input_id, expected",
    [
        # Test with ObjectId instance
        (ObjectId("507f1f77bcf86cd799439011"), ObjectId("507f1f77bcf86cd799439011")),
        # Test with valid ObjectId string
        ("507f1f77bcf86cd799439011", ObjectId("507f1f77bcf86cd799439011")),
        # Test with invalid ObjectId string (wrong length)
    ],
)
def test_get_object_id(input_id, expected):
    result = get_object_id(input_id)
    assert result == expected
