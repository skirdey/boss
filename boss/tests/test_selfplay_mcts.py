import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from bson import ObjectId

import pytest_asyncio

# Import the classes you want to test from your module.
# Example: from myapp.selfplay_mcts import SelfPlayMCTS, TreeNode, StepGenerationResponse, AgentSelectionAnalysis
from boss.self_play import (
    SelfPlayMCTS,
    TreeNode,
    StepGenerationResponse,
    EstimatedNode,
    AgentSelectionAnalysis,
    TaskEvaluationResponse,
)


@pytest_asyncio.fixture
async def mock_task_queue():
    """
    A mock asyncio.Queue that will yield mock tasks.
    """
    queue = asyncio.Queue()
    # Put a single mock task into the queue (simulating boss sending a task).
    await queue.put({
        "_id": ObjectId(),
        "description": "Mock Task Description",
        "targets": ["localhost:5000"],
    })
    return queue


@pytest_asyncio.fixture
async def mock_result_queue():
    """
    A mock asyncio.Queue that will yield mock results.
    """
    queue = asyncio.Queue()
    # We'll populate this queue in specific tests if needed
    return queue


@pytest_asyncio.fixture
async def mock_selfplay_response_queue():
    """
    A mock asyncio.Queue that receives the step/agent selection from MCTS -> BOSS
    """
    return asyncio.Queue()


@pytest_asyncio.fixture
async def mcts_instance(mock_task_queue, mock_result_queue, mock_selfplay_response_queue):
    """
    Create an instance of SelfPlayMCTS with mocked queues and a fake Mongo URI.
    """
    mcts = SelfPlayMCTS(
        solutions_path="test_mcts_solutions.json",
        task_queue=mock_task_queue,
        result_queue=mock_result_queue,
        selfplay_response_queue=mock_selfplay_response_queue,
        mongo_uri="mongodb://fakehost:27017",
    )

    # Patch motor client to avoid real DB interactions:
    mcts.mongo_client = AsyncMock()
    mcts.db = AsyncMock()
    mcts.tasks_collection = AsyncMock()
    mcts.agents_collection = AsyncMock()
    mcts.task_history_collection = AsyncMock()

    # Also set up some default return values for agent_collection:
    # e.g., let's say we have two agents in the DB
    mcts.agents_collection.find.return_value.to_list = AsyncMock(
        return_value=[
            {"agent_id": "agent_1", "capabilities": ["capability_a", "capability_b"]},
            {"agent_id": "agent_2", "capabilities": ["capability_b", "capability_c"]},
        ]
    )
    return mcts


@pytest.mark.asyncio
async def test_from_boss_tasks(mcts_instance):
    """
    Test that from_boss_tasks() reads from the task_queue,
    then calls select_step_and_agent() to handle the new task.
    """
    with patch.object(mcts_instance, "select_step_and_agent", new=AsyncMock()) as mock_select:
        # We'll run from_boss_tasks() in a background task so it doesn't block forever
        task = asyncio.create_task(mcts_instance.from_boss_tasks())

        # Let the consumer run one iteration
        await asyncio.sleep(0.1)

        # Cancel the infinite loop
        task.cancel()

        mock_select.assert_awaited_once()


@pytest.mark.asyncio
async def test_select_step_and_agent_generates_initial_steps(mcts_instance):
    """
    Test that select_step_and_agent() calls _generate_initial_steps
    if the node has no untried steps and no children.
    """
    mock_task = {
        "_id": ObjectId(),
        "description": "Test task for step generation",
        "targets": [],
    }

    # Mock out the LLM step generation
    mock_response = StepGenerationResponse(
        overall_plan="Mock Plan",
        estimated_steps=[
            EstimatedNode(step_description="Step 1"),
            EstimatedNode(step_description="Step 2"),
        ],
    )

    with patch.object(
        mcts_instance, "_generate_initial_steps", return_value=mock_response
    ) as mock_gen_steps, patch.object(
        mcts_instance, "_tree_policy", new=AsyncMock()
    ) as mock_tree_policy, patch.object(
        mcts_instance, "update_task_tree_structure", new=AsyncMock()
    ):
        await mcts_instance.select_step_and_agent(mock_task)

        # We expect the root node to have untried steps assigned
        root_node = mcts_instance.root_nodes[str(mock_task["_id"])]
        assert len(root_node.untried_steps) == 2
        mock_gen_steps.assert_called_once()


@pytest.mark.asyncio
async def test_generate_initial_steps_calls_generate_steps_from_description(mcts_instance):
    """
    Ensure that _generate_initial_steps internally calls _generate_steps_from_description
    and returns the correct StepGenerationResponse.
    """
    mock_task = {
        "_id": ObjectId(),
        "description": "Some task description",
    }

    # Prepare a fake StepGenerationResponse
    expected_response = StepGenerationResponse(
        overall_plan="A Plan",
        estimated_steps=[EstimatedNode(step_description="Mock Step")],
    )

    # Patch the `_generate_steps_from_description` method to return our expected response
    with patch.object(
        mcts_instance, "_generate_steps_from_description", return_value=expected_response
    ) as mock_gen_desc:
        result = await mcts_instance._generate_initial_steps(
            mock_task, capabilities_list="Agent agent_1: c1, c2"
        )

        assert result == expected_response
        mock_gen_desc.assert_awaited_once()


@pytest.mark.asyncio
async def test_from_boss_results_evaluates_and_backpropagates(mcts_instance, mock_result_queue):
    """
    Test from_boss_results() to ensure that when a result is put on the queue,
    it calls _evaluate_agent_performance and _backpropagate.
    """
    # Create a mock node in the tree
    import uuid
    task_id = str(ObjectId())
    step_id = str(ObjectId())
    node = TreeNode(
        task_id=task_id,
        step_id=step_id,
        step_description="Some Step",
    )
    mcts_instance.root_nodes[task_id] = node

    # Put a result in the queue
    result = {
        "task_id": task_id,
        "step_id": step_id,
        "agent_output": "Some agent output",
        "metadata": {"foo": "bar"},
        "step_description": "Some Step",
        "task_description": "Overall Task Desc",
        "agent_id": "agent_1",
    }
    await mock_result_queue.put(result)

    with patch.object(
        mcts_instance, "_evaluate_agent_performance", return_value=TaskEvaluationResponse(
            success=True,
            confidence_score=0.7,
            reasoning="Mock reasoning",
            explanation="Mock explanation",
            critique="Mock critique",
            agent_output="Mock agent output",
        )
    ) as mock_eval, patch.object(
        mcts_instance, "_backpropagate", new=AsyncMock()
    ) as mock_backprop, patch.object(
        mcts_instance, "update_task_tree_structure", new=AsyncMock()
    ):
        # Run from_boss_results in the background
        consumer_task = asyncio.create_task(mcts_instance.from_boss_results())

        # Wait a bit for the consumer to process the queue
        await asyncio.sleep(0.1)

        # Cancel the infinite loop
        consumer_task.cancel()

        # Check that evaluation was called
        mock_eval.assert_called_once()
        # Check that backprop was called
        mock_backprop.assert_awaited_once()
        # Verify that the node's llm_evaluation was updated
        assert node.llm_evaluation["success"] is True


@pytest.mark.asyncio
async def test_simulate_sends_to_selfplay_response_queue(mcts_instance, mock_selfplay_response_queue):
    """
    Test that _simulate() puts the correct execution_request into selfplay_response_queue.
    """
    # Create a parent node for context

    task_id = str(ObjectId())
    step_id = str(ObjectId())

    parent_node = TreeNode(
        task_id=task_id,
        step_id=step_id,
        step_description="Parent Step",
    )
    # Create a child node that we will simulate
    node = TreeNode(
        task_id=task_id,
        step_id=step_id,
        step_description="Child Step",
        agent_id="agent_1",
        parent=parent_node,
    )

    # Fake task dict
    task = {
        "_id": task_id,
        "description": "Task description",
        "targets": ["localhost:8080"]
    }

    await mcts_instance._simulate(node, task)

    # The queue should now have exactly one item
    assert not mock_selfplay_response_queue.empty()
    request = await mock_selfplay_response_queue.get()

    assert request["task_id"] == node.task_id
    assert request["step_id"] == node.step_id
    assert request["step_description"] == node.step_description
    assert request["agent_id"] == node.agent_id
    assert request["task_description"] == task["description"]


# @pytest.mark.asyncio
# async def test_from_boss_results_multiple_results(mcts_instance, mock_result_queue):
#     """Test from_boss_results() with multiple results in the queue."""
#     task_id = str(ObjectId())
#     # Add a mock task to the task_store
#     mcts_instance.task_store[task_id] = {"_id": task_id, "description": "Mock Task"}

#     # Create a root node
#     root_node = TreeNode(task_id=task_id)
#     mcts_instance.root_nodes[task_id] = root_node

#     node1 = TreeNode(task_id=task_id, step_id=str(ObjectId()), step_description="Step 1")
#     node2 = TreeNode(task_id=task_id, step_id=str(ObjectId()), step_description="Step 2")

#     # Add nodes as children of the root (assuming this is the correct tree structure)
#     root_node.add_child(node1)
#     root_node.add_child(node2)

#     await mock_result_queue.put({"task_id": task_id, "step_id": node1.step_id, "task_description": "Mock Task"})
#     await mock_result_queue.put({"task_id": task_id, "step_id": node2.step_id, "task_description": "Mock Task"})

#     with patch.object(
#         mcts_instance, "_evaluate_agent_performance", AsyncMock(return_value=TaskEvaluationResponse(success=True, confidence_score=0.7, reasoning="Mock", explanation="Mock", critique="Mock", agent_output="Mock"))
#     ) as mock_eval, patch.object(
#         mcts_instance, "_backpropagate", new=AsyncMock()
#     ) as mock_backprop, patch.object(
#         mcts_instance, "update_task_tree_structure", new=AsyncMock()
#     ) as mock_update_tree:
#         consumer_task = asyncio.create_task(mcts_instance.from_boss_results())
#         # Wait until the queue is empty
#         while not mock_result_queue.empty():
#             await asyncio.sleep(0.01)
#         consumer_task.cancel()
#         assert mock_eval.call_count == 2
#         assert mock_backprop.await_count == 2
#         assert mock_update_tree.await_count == 2


@pytest.mark.asyncio
async def test_uct_select(mcts_instance):
    """
    Test the _uct_select method, ensuring that the child with the highest UCT score is selected.
    """

    task_id = str(ObjectId())
    root = TreeNode(task_id=task_id)
    child1 = TreeNode(task_id=task_id, step_description="Step1")
    child2 = TreeNode(task_id=task_id, step_description="Step2")

    # Manually set visits/value to check the exploitation term
    child1.visits = 2
    child1.value = 2.0   # exploitation = 1.0
    child2.visits = 2
    child2.value = 1.0   # exploitation = 0.5

    root.add_child(child1)
    root.add_child(child2)

    # The UCB formula includes an exploration term, but for deterministic testing we can 
    # just mock out parent's visits to control the log parent visits piece
    root.visits = 10

    selected = mcts_instance._uct_select(root)
    # Because child1 has a better exploitation (1.0 vs 0.5),
    # we expect child1 to be selected unless exploration drastically changes it.
    # In practice, child1 is likely to have the higher UCT.
    assert selected == child1


@pytest.mark.asyncio
async def test_backpropagate_updates_value_and_visits(mcts_instance):
    """
    Test that _backpropagate properly updates visits and values up the chain.
    """

    task_id = str(ObjectId())
    root = TreeNode(task_id=task_id, step_description="root")
    child = TreeNode(task_id=task_id, step_description="child", parent=root)
    grandchild = TreeNode(task_id=task_id, step_description="grandchild", parent=child)

    # Manually link them
    root.add_child(child)
    child.add_child(grandchild)

    # Let's backprop from grandchild with a reward=1.0
    await mcts_instance._backpropagate(grandchild, 1.0)

    # Check visits
    assert root.visits == 1
    assert child.visits == 1
    assert grandchild.visits == 1

    # Check values
    assert root.value == 1.0
    assert child.value == 1.0
    assert grandchild.value == 1.0


@pytest.mark.asyncio
async def test_initialize_child_untried_steps(mcts_instance):
    """
    Test that _initialize_child_untried_steps populates untried_steps for the child.
    """
    # Put a mock task into the store
    test_task_id = str(ObjectId())
    mcts_instance.task_store[test_task_id] = {
        "_id": test_task_id,
        "description": "Task description for child init",
    }

    # Mock child node
    parent_node = TreeNode(
        task_id=test_task_id,
        step_description="Parent Step",
    )
    child_node = TreeNode(
        task_id=test_task_id,
        step_description="Child Step",
        parent=parent_node,
    )

    # Patch the step generation
    mock_steps = StepGenerationResponse(
        estimated_steps=[
            EstimatedNode(step_description="Generated Step 1"),
            EstimatedNode(step_description="Generated Step 2"),
        ]
    )
    with patch.object(
        mcts_instance,
        "_generate_steps_from_description",
        new=AsyncMock(return_value=mock_steps),
    ):
        await mcts_instance._initialize_child_untried_steps(child_node)

        # The child should now have 2 untried steps
        assert len(child_node.untried_steps) == 2
        assert child_node.untried_steps[0].step_description == "Generated Step 1"
        assert child_node.untried_steps[1].step_description == "Generated Step 2"



@pytest.mark.asyncio
async def test_backpropagate_zero_reward(mcts_instance):
    """Test _backpropagate with a reward of 0.0."""

    task_id = str(ObjectId())
    root = TreeNode(task_id=task_id, step_description="root")
    child = TreeNode(task_id=task_id, step_description="child", parent=root)
    root.add_child(child)
    await mcts_instance._backpropagate(child, 0.0)
    assert root.visits == 1
    assert child.visits == 1
    assert root.value == 0.0
    assert child.value == 0.0


@pytest.mark.asyncio
async def test_backpropagate_negative_reward(mcts_instance):
    """Test _backpropagate with a negative reward."""

    task_id = str(ObjectId())
    root = TreeNode(task_id=task_id, step_description="root")
    child = TreeNode(task_id=task_id, step_description="child", parent=root)
    root.add_child(child)
    await mcts_instance._backpropagate(child, -0.5)
    assert root.visits == 1
    assert child.visits == 1
    assert root.value == -0.5
    assert child.value == -0.5


@pytest.mark.asyncio
async def test_initialize_child_untried_steps_task_not_found(mcts_instance):
    """Test _initialize_child_untried_steps when the task is not found."""
    task_id = str(ObjectId())
    child_node = TreeNode(task_id=task_id, step_description="Child Step")
    with patch.object(mcts_instance, "_generate_steps_from_description", new=AsyncMock()) as mock_gen:
        await mcts_instance._initialize_child_untried_steps(child_node)
        assert not child_node.untried_steps
        mock_gen.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_child_untried_steps_empty_task_description(mcts_instance):
    """Test _initialize_child_untried_steps with an empty task description."""
    test_task_id = str(ObjectId())
    mcts_instance.task_store[test_task_id] = {"_id": test_task_id, "description": ""}
    parent_node = TreeNode(task_id=test_task_id, step_description="Parent Step")
    child_node = TreeNode(task_id=test_task_id, step_description="Child Step", parent=parent_node)
    with patch.object(mcts_instance, "_generate_steps_from_description", new=AsyncMock()) as mock_gen:
        await mcts_instance._initialize_child_untried_steps(child_node)
        assert not child_node.untried_steps
        mock_gen.assert_not_called()


@pytest.mark.asyncio
async def test_simulate_node_targets_override_task_targets(mcts_instance, mock_selfplay_response_queue):
    """Test _simulate when node has specific targets, overriding task targets."""

    task_id = str(ObjectId())
    node = TreeNode(
        task_id=task_id,
        step_id=str(ObjectId()),
        step_description="Child Step",
        agent_id="agent_1",
        targets=["custom_target:9000"],
    )
    task = {"_id": task_id, "description": "Task description", "targets": ["task_target:8080"]}
    await mcts_instance._simulate(node, task)
    request = await mock_selfplay_response_queue.get()
    assert request["targets"] == ["custom_target:9000"]


@pytest.mark.asyncio
async def test_simulate_missing_agent_id(mcts_instance, mock_selfplay_response_queue):
    """Test _simulate with a node missing agent_id."""

    task_id = str(ObjectId())
    node = TreeNode(task_id=task_id, step_id=str(ObjectId()), step_description="Some Step")
    task = {"_id": task_id, "description": "Task description", "targets": []}
    await mcts_instance._simulate(node, task)
    assert mock_selfplay_response_queue.empty()


@pytest.mark.asyncio
async def test_simulate_missing_step_description(mcts_instance, mock_selfplay_response_queue):
    """Test _simulate with a node missing step_description."""

    task_id = str(ObjectId())
    node = TreeNode(task_id=task_id, step_id=str(ObjectId()), agent_id="agent_1")
    task = {"_id": task_id, "description": "Task description", "targets": []}
    await mcts_instance._simulate(node, task)
    assert mock_selfplay_response_queue.empty()
