# tests/test_boss.py

import json
from datetime import datetime, timezone
from unittest.mock import ANY, MagicMock, patch

import pytest
from bson import ObjectId
from pydantic import BaseModel

# Sample data for mocking
sample_task_id = "64b7f2f4f1a4b5d6e7f8a1b2"  # Valid ObjectId
sample_task = {
    "_id": ObjectId(sample_task_id),
    "description": "Sample task description",
    "status": "created",
    "steps": [],
    "current_step_index": 0,  # Set to 0 for tests that require it
}
sample_agent = {
    "agent_id": "agent_1",
    "capabilities": ["capability_a", "capability_b"],
    "status": "active",
}


# Define mock response models if they are Pydantic models
class MockStepEstimationResponse(BaseModel):
    estimated_steps: list


class MockTaskEvaluationResponse(BaseModel):
    success: bool


class MockAgentSelectionAnalysis(BaseModel):
    agent_id: str
    overall_match_score: float


# Define a mock function to serialize tasks if necessary
def mock_serialize_task(task):
    return json.dumps(task).encode("utf-8")


def test_boss_initialization():
    """
    Test the initialization of the BOSS class.
    """
    # Apply patches before importing BOSS
    with patch("boss.MongoClient") as mock_mongo_client, patch(
        "boss.KafkaProducer"
    ) as mock_kafka_producer, patch("boss.KafkaConsumer") as mock_kafka_consumer, patch(
        "boss.OpenAI"
    ) as mock_openai_client:
        # Setup MongoDB mock
        mock_db = MagicMock()
        mock_mongo_client.return_value = mock_db
        # Mock database and collections
        mock_db.__getitem__.side_effect = lambda name: mock_db
        mock_db["task_db"] = mock_db
        mock_db["tasks"] = mock_db
        mock_db["agents"].find.return_value = [sample_agent]
        mock_db["task_history"] = mock_db

        # Mock KafkaProducer
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        # Mock KafkaConsumer
        mock_consumer_instance = MagicMock()
        mock_consumer_instance.__iter__.return_value = iter([])  # No messages
        mock_kafka_consumer.return_value = mock_consumer_instance

        # Mock OpenAI
        mock_openai_instance = MagicMock()
        mock_openai_client.return_value = mock_openai_instance

        # Import BOSS after patches are applied
        from boss import BOSS

        # Initialize BOSS
        boss = BOSS(
            db_uri="mongodb://test_db_uri/",
            kafka_bootstrap_servers="test_kafka:9092",
            openai_api_key="test_api_key",
        )

        # Assert MongoClient was called with the correct URI
        mock_mongo_client.assert_called_with(
            "mongodb://test_db_uri/", serverSelectionTimeoutMS=5000
        )
        mock_db.server_info.assert_called()

        # Assert KafkaProducer was initialized correctly
        mock_kafka_producer.assert_called_with(
            bootstrap_servers="test_kafka:9092",
            value_serializer=ANY,  # Use ANY for callables
        )
        # Optionally, check that value_serializer is callable
        assert callable(boss.producer.value_serializer)

        # Assert KafkaConsumer was initialized correctly
        mock_kafka_consumer.assert_called_with(
            bootstrap_servers="test_kafka:9092",
            value_deserializer=ANY,  # Use ANY for callables
            consumer_timeout_ms=1000,
            group_id="boss_group",
        )
        # Optionally, check that value_deserializer is callable
        assert callable(boss.result_consumer.value_deserializer)

        # Assert OpenAI was initialized with the correct API key
        mock_openai_client.assert_called_with(api_key="test_api_key")


def test_handle_agent_result_failure():
    """
    Test the handle_agent_result method when the step evaluation fails.
    """
    with patch("boss.MongoClient") as mock_mongo_client, patch(
        "boss.KafkaProducer"
    ) as mock_kafka_producer, patch("boss.KafkaConsumer") as mock_kafka_consumer, patch(
        "boss.OpenAI"
    ) as mock_openai_client:
        # Setup MongoDB mock
        mock_db = MagicMock()
        mock_mongo_client.return_value = mock_db
        mock_db.__getitem__.side_effect = lambda name: mock_db
        # Update current_step_index to a valid value (e.g., 0)
        task_with_step = sample_task.copy()
        task_with_step["current_step_index"] = 0
        task_with_step["steps"] = [
            {
                "step_number": 1,
                "step_description": "Initial Step",
                "state": "created",
                "estimated_duration": 10,
                "confidence_score": 0.9,
                "expected_outcome": "Outcome 1",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        ]
        mock_db["tasks"].find_one.return_value = task_with_step

        # Mock KafkaProducer and KafkaConsumer
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        mock_consumer_instance = MagicMock()
        mock_consumer_instance.__iter__.return_value = iter([])  # No messages
        mock_kafka_consumer.return_value = mock_consumer_instance

        # Mock OpenAI
        mock_openai_instance = MagicMock()
        mock_openai_client.return_value = mock_openai_instance

        # Import BOSS after patches
        from boss import BOSS

        # Initialize BOSS
        boss = BOSS(
            db_uri="mongodb://test_db_uri/",
            kafka_bootstrap_servers="test_kafka:9092",
            openai_api_key="test_api_key",
        )

        # Mock OpenAI evaluation to return failure
        evaluation_response = MockTaskEvaluationResponse(success=False)
        boss.call_openai_api_structured = MagicMock(return_value=evaluation_response)

        # Mock task update
        mock_db["tasks"].update_one = MagicMock()

        # Mock _process_task_with_inference and _handle_failed_completion
        boss._process_task_with_inference = MagicMock()
        boss._handle_failed_completion = MagicMock()

        # Execute
        result = {
            "task_id": sample_task_id,
            "result": "Agent result that does NOT satisfy the step.",
        }
        boss.handle_agent_result(result)

        # Assert OpenAI evaluation was called
        boss.call_openai_api_structured.assert_called_once()

        # Assert task was updated to mark the step as failed
        mock_db["tasks"].update_one.assert_called_once()
        update_filter, update_values = mock_db["tasks"].update_one.call_args[0]
        assert update_filter == {"_id": ObjectId(sample_task_id)}
        assert "steps" in update_values["$set"]
        assert update_values["$set"]["current_step_index"] is None
        assert update_values["$set"]["current_agent"] is None

        # Assert _process_task_with_inference was not called
        boss._process_task_with_inference.assert_not_called()

        # Assert _handle_failed_completion was called
        boss._handle_failed_completion.assert_called_once()


def test_call_openai_api_structured_success():
    """
    Test the call_openai_api_structured method for a successful API call.
    """
    with patch("boss.MongoClient") as mock_mongo_client, patch(
        "boss.KafkaProducer"
    ) as mock_kafka_producer, patch("boss.KafkaConsumer") as mock_kafka_consumer, patch(
        "boss.OpenAI"
    ) as mock_openai_client:
        # Setup MongoDB mock
        mock_db = MagicMock()
        mock_mongo_client.return_value = mock_db
        mock_db.__getitem__.side_effect = lambda name: mock_db
        mock_db["agents"].find.return_value = [sample_agent]

        # Mock KafkaProducer and KafkaConsumer
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        mock_consumer_instance = MagicMock()
        mock_consumer_instance.__iter__.return_value = iter([])  # No messages
        mock_kafka_consumer.return_value = mock_consumer_instance

        # Mock OpenAI
        mock_openai_instance = MagicMock()
        mock_openai_client.return_value = mock_openai_instance

        # Import BOSS after patches
        from boss import BOSS

        # Initialize BOSS
        boss = BOSS(
            db_uri="mongodb://test_db_uri/",
            kafka_bootstrap_servers="test_kafka:9092",
            openai_api_key="test_api_key",
        )

        # Mock OpenAI response
        response_json = '{"success": true}'
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=response_json))]
        boss.openai_client.beta.chat.completions.parse.return_value = mock_completion

        # Define a mock response model
        class MockResponseModel(BaseModel):
            success: bool

        # Execute
        response = boss.call_openai_api_structured(
            messages=[{"role": "user", "content": "Test message"}],
            response_model=MockResponseModel,
            model="gpt-4o-mini",
        )

        # Assert OpenAI API was called with correct parameters
        boss.openai_client.beta.chat.completions.parse.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0,
            response_format=MockResponseModel,
            max_tokens=2000,
        )

        # Assert the response is parsed correctly
        assert isinstance(response, MockResponseModel)
        assert response.success is True


def test_call_openai_api_structured_failure():
    """
    Test the call_openai_api_structured method when the API call fails.
    """
    with patch("boss.MongoClient") as mock_mongo_client, patch(
        "boss.KafkaProducer"
    ) as mock_kafka_producer, patch("boss.KafkaConsumer") as mock_kafka_consumer, patch(
        "boss.OpenAI"
    ) as mock_openai_client:
        # Setup MongoDB mock
        mock_db = MagicMock()
        mock_mongo_client.return_value = mock_db
        mock_db.__getitem__.side_effect = lambda name: mock_db
        mock_db["agents"].find.return_value = [sample_agent]

        # Mock KafkaProducer and KafkaConsumer
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        mock_consumer_instance = MagicMock()
        mock_consumer_instance.__iter__.return_value = iter([])  # No messages
        mock_kafka_consumer.return_value = mock_consumer_instance

        # Mock OpenAI
        mock_openai_instance = MagicMock()
        mock_openai_client.return_value = mock_openai_instance

        # Import BOSS after patches
        from boss import BOSS

        # Initialize BOSS
        boss = BOSS(
            db_uri="mongodb://test_db_uri/",
            kafka_bootstrap_servers="test_kafka:9092",
            openai_api_key="test_api_key",
        )

        # Simulate an exception during API call
        boss.openai_client.beta.chat.completions.parse.side_effect = Exception(
            "API error"
        )

        # Define a mock response model
        class MockResponseModel(BaseModel):
            success: bool

        # Execute & Assert
        with pytest.raises(Exception) as exc_info:
            boss.call_openai_api_structured(
                messages=[{"role": "user", "content": "Test message"}],
                response_model=MockResponseModel,
                model="gpt-4o-mini",
            )
        assert "API error" in str(exc_info.value)

        # Assert OpenAI API was called
        boss.openai_client.beta.chat.completions.parse.assert_called_once()