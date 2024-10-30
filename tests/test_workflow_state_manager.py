import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from bson import ObjectId

from models import StepResult, TaskState
from workflow_state_manager import WorkflowStateManager


@pytest.fixture
def mock_collections():
    tasks_collection = Mock()
    task_history_collection = Mock()
    return tasks_collection, task_history_collection


@pytest.fixture
def workflow_manager(mock_collections):
    tasks_collection, task_history_collection = mock_collections
    return WorkflowStateManager(tasks_collection, task_history_collection)


@pytest.fixture
def sample_task():
    return {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "workflow_state": {
            "completed_steps": [],
            "remaining_steps": [
                {"step_description": "Step 1", "expected_outcome": "Result 1"},
                {"step_description": "Step 2", "expected_outcome": "Result 2"},
            ],
            "current_agent": None,
            "last_inference": datetime.now(timezone.utc),
        },
        "current_step": 0,
        "status": TaskState.PENDING_NEXT_STEP,
        "step_estimation": {
            "estimated_steps": [
                {"step_description": "Step 1", "expected_outcome": "Result 1"},
                {"step_description": "Step 2", "expected_outcome": "Result 2"},
            ]
        },
    }


# Test get_current_step
def test_get_current_step_success(workflow_manager, mock_collections, sample_task):
    tasks_collection, _ = mock_collections
    tasks_collection.find_one.return_value = sample_task

    current_step = workflow_manager.get_current_step("507f1f77bcf86cd799439011")

    assert current_step == sample_task["workflow_state"]["remaining_steps"][0]
    tasks_collection.find_one.assert_called_once()


def test_get_current_step_nonexistent_task(workflow_manager, mock_collections):
    tasks_collection, _ = mock_collections
    tasks_collection.find_one.return_value = None

    current_step = workflow_manager.get_current_step("507f1f77bcf86cd799439011")

    assert current_step is None


def test_get_current_step_invalid_index(
    workflow_manager, mock_collections, sample_task
):
    tasks_collection, _ = mock_collections
    modified_task = sample_task.copy()
    modified_task["current_step"] = 5
    tasks_collection.find_one.return_value = modified_task

    current_step = workflow_manager.get_current_step("507f1f77bcf86cd799439011")

    assert current_step is None


# Test update_workflow_state
def test_update_workflow_state_success(workflow_manager, mock_collections, sample_task):
    tasks_collection, _ = mock_collections
    tasks_collection.find_one.return_value = sample_task
    tasks_collection.update_one.return_value = Mock(modified_count=1)

    step_result = StepResult(
        step_description="Step 1",
        result="Completed successfully",
        success=True,
        execution_time=1.5,
        error=None,
        metadata={"additional": "info"},
    )

    result = workflow_manager.update_workflow_state(
        "507f1f77bcf86cd799439011", step_result, "agent123"
    )

    assert result is True
    tasks_collection.update_one.assert_called_once()
    update_call = tasks_collection.update_one.call_args[0]
    assert "workflow_state" in update_call[1]["$set"]
    assert update_call[1]["$set"]["status"] == TaskState.PENDING_NEXT_STEP
    assert len(update_call[1]["$set"]["workflow_state"]["completed_steps"]) == 1


def test_update_workflow_state_step_mismatch(
    workflow_manager, mock_collections, sample_task
):
    tasks_collection, _ = mock_collections
    tasks_collection.find_one.return_value = sample_task

    step_result = StepResult(
        step_description="Wrong Step",
        result="Completed",
        success=True,
        execution_time=1.0,
        error=None,
        metadata={},
    )

    result = workflow_manager.update_workflow_state(
        "507f1f77bcf86cd799439011", step_result, "agent123"
    )

    assert result is False
    tasks_collection.update_one.assert_not_called()


def test_update_workflow_state_failed_step(
    workflow_manager, mock_collections, sample_task
):
    tasks_collection, _ = mock_collections
    tasks_collection.find_one.return_value = sample_task
    tasks_collection.update_one.return_value = Mock(modified_count=1)

    step_result = StepResult(
        step_description="Step 1",
        result=None,
        success=False,
        execution_time=1.0,
        error="Error occurred",
        metadata={},
    )

    result = workflow_manager.update_workflow_state(
        "507f1f77bcf86cd799439011", step_result, "agent123"
    )

    assert result is True
    update_call = tasks_collection.update_one.call_args[0]
    assert update_call[1]["$set"]["status"] == TaskState.FAILED
    assert "error" in update_call[1]["$set"]


def test_update_workflow_state_final_step(
    workflow_manager, mock_collections, sample_task
):
    tasks_collection, _ = mock_collections
    modified_task = sample_task.copy()
    modified_task["workflow_state"]["remaining_steps"] = [
        {"step_description": "Final Step", "expected_outcome": "Result"}
    ]
    tasks_collection.find_one.return_value = modified_task
    tasks_collection.update_one.return_value = Mock(modified_count=1)

    step_result = StepResult(
        step_description="Final Step",
        result="Completed",
        success=True,
        execution_time=1.0,
        error=None,
        metadata={},
    )

    result = workflow_manager.update_workflow_state(
        "507f1f77bcf86cd799439011", step_result, "agent123"
    )

    assert result is True
    update_call = tasks_collection.update_one.call_args[0]
    assert update_call[1]["$set"]["status"] == TaskState.COMPLETED_WORKFLOW


# Test add_step
def test_add_step_success(workflow_manager, mock_collections):
    tasks_collection, _ = mock_collections
    tasks_collection.update_one.return_value = Mock(modified_count=1)

    new_step = {"step_description": "New Step", "expected_outcome": "Result"}
    result = workflow_manager.add_step("507f1f77bcf86cd799439011", new_step)

    assert result is True
    tasks_collection.update_one.assert_called_once()


def test_add_step_failure(workflow_manager, mock_collections):
    tasks_collection, _ = mock_collections
    tasks_collection.update_one.return_value = Mock(modified_count=0)

    new_step = {"step_description": "New Step", "expected_outcome": "Result"}
    result = workflow_manager.add_step("507f1f77bcf86cd799439011", new_step)

    assert result is False
