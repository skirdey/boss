import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch

import pytest
from bson import ObjectId

from agent_scheduler import AgentScheduler  # Update import path as needed
from models import TaskState


@pytest.fixture
def mock_collections():
    tasks_collection = Mock()
    agents_collection = Mock()
    return tasks_collection, agents_collection


@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock_openai_cls:
        # Create a mock instance that will be returned when OpenAI is instantiated
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Mock the beta.chat.completions.parse method
        mock_client.beta.chat.completions.parse = Mock()

        yield mock_client


@pytest.fixture
def scheduler(mock_collections, mock_openai):
    tasks_collection, agents_collection = mock_collections
    with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy_key"}):
        scheduler = AgentScheduler(tasks_collection, agents_collection)
        scheduler.openai_client = mock_openai  # Replace the client with our mock
    return scheduler


@pytest.fixture
def sample_task():
    return {
        "_id": ObjectId(),
        "status": TaskState.CREATED.value,
        "description": "Test task",
    }


@pytest.fixture
def sample_agents():
    return [
        {
            "agent_id": "agent1",
            "status": "active",
            "current_tasks": 2,
            "capabilities": ["capability1", "capability2"],
        },
        {
            "agent_id": "agent2",
            "status": "active",
            "current_tasks": 1,
            "capabilities": ["capability1", "capability3"],
        },
    ]


class TestAgentScheduler:
    def test_get_object_id_valid(self, scheduler):
        valid_id = str(ObjectId())
        result = scheduler._get_object_id(valid_id)
        assert isinstance(result, ObjectId)
        assert str(result) == valid_id

    def test_get_object_id_invalid(self, scheduler):
        result = scheduler._get_object_id("invalid_id")
        assert result is None

    def test_evaluate_agent_capability_success(self, scheduler, mock_openai):
        # Mock OpenAI response
        mock_completion = Mock()
        mock_completion.choices = [
            Mock(
                message=Mock(
                    parsed=Mock(
                        can_handle=True,
                        confidence=0.8,
                        reasoning="Agent has required capabilities",
                    )
                )
            )
        ]
        mock_openai.beta.chat.completions.parse.return_value = mock_completion

        can_handle, confidence = scheduler.evaluate_agent_capability(
            "Test task", ["capability1", "capability2"]
        )

        assert can_handle is True
        assert confidence == 0.8

    def test_get_agent_load_success(self, scheduler, mock_collections):
        tasks_collection, agents_collection = mock_collections
        agents_collection.find_one.return_value = {"current_tasks": 3}

        load = scheduler.get_agent_load("agent1")
        assert load == 3
        agents_collection.find_one.assert_called_once()

    def test_get_agent_load_not_found(self, scheduler, mock_collections):
        tasks_collection, agents_collection = mock_collections
        agents_collection.find_one.return_value = None

        load = scheduler.get_agent_load("nonexistent_agent")
        assert load == 0

    def test_update_agent_task_count_success(self, scheduler, mock_collections):
        tasks_collection, agents_collection = mock_collections
        agents_collection.update_one.return_value = Mock(modified_count=1)

        result = scheduler.update_agent_task_count("agent1", 1)
        assert result is True
        agents_collection.update_one.assert_called_once()

    def test_update_agent_task_count_failure(self, scheduler, mock_collections):
        tasks_collection, agents_collection = mock_collections
        agents_collection.update_one.return_value = Mock(modified_count=0)

        result = scheduler.update_agent_task_count("agent1", 1)
        assert result is False

    def test_assign_task_success(
        self, scheduler, mock_collections, sample_task, sample_agents
    ):
        tasks_collection, agents_collection = mock_collections

        # Mock database responses
        agents_collection.find.return_value = sample_agents
        tasks_collection.find_one_and_update.return_value = {
            **sample_task,
            "agent_id": "agent1",
            "status": "In_Progress",
        }
        agents_collection.update_one.return_value = Mock(modified_count=1)

        # Mock capability evaluation
        with patch.object(
            scheduler, "evaluate_agent_capability", return_value=(True, 0.8)
        ):
            result = scheduler.assign_task(
                str(sample_task["_id"]), "Test task description"
            )

        assert result == "agent1"
        tasks_collection.find_one_and_update.assert_called_once()

    def test_assign_task_no_available_agents(self, scheduler, mock_collections):
        tasks_collection, agents_collection = mock_collections
        agents_collection.find.return_value = []

        result = scheduler.assign_task(str(ObjectId()), "Test task")
        assert result is None

    def test_assign_task_invalid_id(self, scheduler):
        result = scheduler.assign_task("invalid_id", "Test task")
        assert result is None

    def test_assign_task_rollback_on_agent_update_failure(
        self, scheduler, mock_collections, sample_task, sample_agents
    ):
        tasks_collection, agents_collection = mock_collections

        # Mock successful task assignment but failed agent update
        agents_collection.find.return_value = sample_agents
        tasks_collection.find_one_and_update.return_value = {
            **sample_task,
            "agent_id": "agent1",
            "status": "In_Progress",
        }
        agents_collection.update_one.return_value = Mock(modified_count=0)

        # Mock capability evaluation
        with patch.object(
            scheduler, "evaluate_agent_capability", return_value=(True, 0.8)
        ):
            result = scheduler.assign_task(
                str(sample_task["_id"]), "Test task description"
            )

        assert result is None
        # Verify rollback was attempted
        assert tasks_collection.update_one.called

    def test_assign_task_already_assigned(
        self, scheduler, mock_collections, sample_task, sample_agents
    ):
        tasks_collection, agents_collection = mock_collections

        # Mock task already assigned
        agents_collection.find.return_value = sample_agents
        tasks_collection.find_one_and_update.return_value = None

        with patch.object(
            scheduler, "evaluate_agent_capability", return_value=(True, 0.8)
        ):
            result = scheduler.assign_task(
                str(sample_task["_id"]), "Test task description"
            )

        assert result is None

    def test_assign_task_no_capable_agents(
        self, scheduler, mock_collections, sample_agents
    ):
        tasks_collection, agents_collection = mock_collections
        agents_collection.find.return_value = sample_agents

        # Mock all agents being incapable
        with patch.object(
            scheduler, "evaluate_agent_capability", return_value=(False, 0.0)
        ):
            result = scheduler.assign_task(str(ObjectId()), "Test task")

        assert result is None
