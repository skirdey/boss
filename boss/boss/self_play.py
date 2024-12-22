import asyncio
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import motor.motor_asyncio
from bson import ObjectId
from pydantic import BaseModel, Field

from boss.boss_prompts import BossPrompts
from boss.llm_client import call_openai_api_structured
from boss.models import TaskEvaluationResponse


class EstimatedNode(BaseModel):
    step_description: str
    expected_outcome: Optional[str] = None
    confidence_score: Optional[float] = None  # Value between 0 and 1
    estimated_duration_minutes: Optional[int] = None
    targets: Optional[List[str]] = Field(
        default=None,
        description="List of targets for this step often in the 'url:port' or 'protocol://url:port' format",
    )


class StepGenerationResponse(BaseModel):
    overall_plan: Optional[str] = None
    estimated_steps: List[EstimatedNode]
    additional_steps: Optional[List[EstimatedNode]] = None  # If any
    critical_path_steps: Optional[List[str]] = None  # List of step descriptions


class AgentSelectionAnalysis(BaseModel):
    agent_id: str
    overall_match_score: int  # A score from 0 to 100
    reasons: str  # Explanation of why the agent is suitable
    suitability: str  # Summary of agent's suitability
    final_recommendation: Optional[str] = None  # New attribute


# Define a custom formatter
class PurpleFormatter(logging.Formatter):
    PURPLE = "\033[95m"  # ANSI code for bright purple
    RESET = "\033[0m"  # ANSI code to reset color

    def format(self, record):
        message = super().format(record)
        return f"{self.PURPLE}{message}{self.RESET}"


# Configure the logger
logger = logging.getLogger("SelfPlayMCTS")
logger.setLevel(logging.INFO)
logger.propagate = False

# Create console handler with the custom formatter
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(
    PurpleFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

# Add the handler to the logger
logger.addHandler(handler)


class TreeNode:
    def __init__(
        self,
        task_id: str,
        step_id: Optional[str] = None,
        step_description: Optional[str] = None,
        agent_id: Optional[str] = None,
        parent: Optional["TreeNode"] = None,
        targets: Optional[List[str]] = None,
    ):
        self.task_id: str = task_id
        self.step_id: str = step_id if step_id else str(ObjectId())
        self.step_description: Optional[str] = step_description
        self.agent_id: Optional[str] = agent_id
        self.visits: int = 0
        self.value: float = 0.0
        self.parent: Optional["TreeNode"] = parent
        self.reasoning: Optional[str] = None

        self.targets: Optional[List[str]] = targets

        self.children: List["TreeNode"] = []
        self.untried_steps: List[EstimatedNode] = []
        self.uct_score: float = 0.0
        self.exploration_history: List[str] = []

        self.llm_evaluation = {
            "score": 0.0,
            "reasoning": "Not evaluated",
            "critique": "No critique",
        }

        self.is_simulation_sent: bool = (
            False  # New attribute to track simulation status
        )

    def __str__(self):
        return (
            f"TreeNode(step_id={self.step_id}, step_description={self.step_description}) "
            f"with {len(self.children)} children \n {self.exploration_history} \n "
            f"{self.llm_evaluation} \n {self.untried_steps}"
        )

    def add_child(self, child_node: "TreeNode"):
        """Add a child node to the current node."""
        self.children.append(child_node)

    def update(self, result: float):
        """Update node statistics."""
        self.visits += 1
        self.value += result

    def fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        return len(self.untried_steps) == 0

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return len(self.children) == 0


class SelfPlayMCTS:
    def __init__(
        self,
        solutions_path: str = "mcts_solutions.json",
        task_queue: asyncio.Queue = None,
        result_queue: asyncio.Queue = None,
        selfplay_response_queue: asyncio.Queue = None,
        mongo_uri: str = "mongodb://localhost:27017",
    ):
        self.solutions_path = Path(solutions_path)
        self.exploration_constant = 1.41  # UCT exploration parameter
        self.agent_details: List[Dict] = []  # Initialize with actual agent details
        self.capabilities_list: str = ""  # Initialize with actual capabilities list

        # Initialize solutions file if it doesn't exist
        if not self.solutions_path.exists():
            self.solutions_path.write_text("[]")

        self.root_nodes: Dict[str, TreeNode] = {}
        self.task_store: Dict[str, Dict] = {}
        self.tree_lock = asyncio.Lock()

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.selfplay_response_queue = selfplay_response_queue

        self.max_experiments: int = 3  # Number of MCTS iterations per task
        self.max_children_per_node: int = 3  # Max number of children per node

        self.pending_simulations: Dict[str, TreeNode] = {}  # Track pending simulations

        self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        self.db = self.mongo_client["task_db"]
        self.tasks_collection = self.db["tasks"]
        self.agents_collection = self.db["agents"]
        self.task_history_collection = self.db["task_history"]

    def get_path_to_node(self, node: TreeNode) -> List[str]:
        """
        Retrieves the path of step descriptions from the root to the given node.

        Returns:
            A list of step descriptions from root to the node.
        """
        path = []
        current = node
        while current.parent is not None:
            if current.step_description:
                path.append(current.step_description)
            current = current.parent
        path.reverse()  # To get the order from root to the node
        return path

    def collect_path_context(self, node: TreeNode) -> str:
        """
        Collects both the step descriptions and relevant LLM evaluations
        from the root to the current node, so the next steps can incorporate
        outcomes of previous steps.
        """
        path_stack = []
        current = node
        while current.parent is not None:
            if current.step_description:
                # Include step description and any LLM evaluation snippet
                eval_str = (
                    f"(Eval: success={current.llm_evaluation.get('success', False)}, "
                    f"score={current.llm_evaluation.get('confidence_score', 0.0)})"
                    if isinstance(current.llm_evaluation, dict)
                    else ""
                )
                path_stack.append(f"Step: {current.step_description} {eval_str}")
            current = current.parent
        path_stack.reverse()
        return "\n".join(path_stack)

    async def start(self):
        self.capabilities_list = "\n".join(
            [
                f"Agent {agent['agent_id']}: {', '.join(agent.get('capabilities', []))}"
                for agent in await self.agents_collection.find().to_list(None)
            ]
        )

        self.agent_details = await self.agents_collection.find().to_list(None)

        asyncio.create_task(self.from_boss_tasks())
        asyncio.create_task(self.from_boss_results())

    async def update_task_tree_structure(self, task: Dict):
        async with self.tree_lock:
            task_id = str(task["_id"])
            root_node = self.root_nodes.get(task_id)
            if not root_node:
                logger.error(f"No root node found for task_id={task_id}")
                return
            # Serialize the tree
            tree_structure = self.serialize_tree(root_node)
            task["tree_structure"] = tree_structure
            # Update task in store or database
            self.task_store[task_id] = task
            await self.update_task_in_database(task)

    def serialize_tree(self, node: TreeNode) -> Dict:
        return {
            "task_id": node.task_id,
            "step_id": node.step_id,
            "step_description": node.step_description,
            "agent_id": node.agent_id,
            "visits": node.visits,
            "value": node.value,
            "children": [self.serialize_tree(child) for child in node.children],
            "llm_evaluation": node.llm_evaluation,
            "uct_score": node.uct_score,
        }

    async def from_boss_results(self):
        """
        Continuously listens for results of executed steps from the result_queue,
        evaluates the outcome, and backpropagates the reward in the MCTS tree.
        """
        while True:
            result = await self.result_queue.get()
            task_id = result["task_id"]
            step_id = result["step_id"]
            # Update the MCTS tree with the result
            node = self._find_node_by_task_and_step_id(task_id, step_id)
            if node:
                # Evaluate the agent's performance using LLM
                evaluation = await self._evaluate_agent_performance(result)
                reward = 1.0 if evaluation.success else 0.0
                # Update node with evaluation details
                node.llm_evaluation = evaluation.model_dump()

                await self._backpropagate(node, reward)
                # Remove from pending simulations
                simulation_key = f"{task_id}_{step_id}"
                self.pending_simulations.pop(simulation_key, None)
                # Update the task tree structure
                task = self.get_task_by_id(task_id)
                if task:
                    await self.update_task_tree_structure(task)
                else:
                    logger.warning(f"Task with id {task_id} not found.")

    async def _evaluate_agent_performance(self, result: Dict) -> TaskEvaluationResponse:
        """
        Use LLM to evaluate how well the agent performed the task.
        """
        try:
            # Get the agent's output
            agent_output = result.get("agent_output", "")
            step_description = result.get("step_description", "")
            task_description = result.get("task_description", "")
            agent_id = result.get("agent_id", "")

            # Prepare the prompt for the LLM
            messages = BossPrompts.format_final_evaluation(
                task_description=task_description,
                step_description=step_description,
                agent_output=agent_output,
                agent_id=agent_id,
            )

            # Call the LLM to get the evaluation
            evaluation = await call_openai_api_structured(
                messages=messages,
                response_model=TaskEvaluationResponse,
                model="gpt-4o-mini",
            )

            if not evaluation:
                logger.error("LLM did not return an evaluation.")
                # Return a default evaluation
                return TaskEvaluationResponse(
                    success=False,
                    confidence_score=1.0,
                    reasoning="LLM did not provide evaluation.",
                    explanation="No evaluation available.",
                    critique="No critique available.",
                )

            evaluation.agent_output = agent_output

            return evaluation

        except Exception as e:
            logger.error(f"Error in evaluating agent performance: {e}")
            # Return a default evaluation
            return TaskEvaluationResponse(
                success=False,
                confidence_score=1.0,
                reasoning="LLM did not provide evaluation.",
                explanation="No evaluation available.",
                critique="No critique available.",
            )

    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        return self.task_store.get(task_id)

    async def from_boss_tasks(self):
        """
        Continuously listens for new tasks. For each incoming task, run MCTS
        to select the next step and best agent, then dispatch it for execution.
        """
        while True:
            task = await self.task_queue.get()

            task_id = str(task["_id"])
            self.task_store[task_id] = task

            logger.info(f"SelfPlay received task: {task.get('_id')}")

            # Run MCTS to select step and agent
            await self.select_step_and_agent(task)

    def _find_node_by_task_and_step_id(
        self, task_id: str, step_id: str
    ) -> Optional[TreeNode]:
        """
        Traverse the MCTS tree to find the node matching the given task_id and step_id.
        """
        root_node = self.root_nodes.get(task_id)
        if not root_node:
            logger.debug(f"No root node found for task_id={task_id}.")
            return None

        queue = [root_node]
        logger.debug(
            f"Starting BFS to find node with task_id={task_id} and step_id={step_id}"
        )

        while queue:
            current_node = queue.pop(0)
            logger.debug(
                f"Visiting node with task_id={current_node.task_id} and step_id={current_node.step_id}"
            )

            if current_node.task_id == task_id and current_node.step_id == step_id:
                logger.info(f"Found node with task_id={task_id} and step_id={step_id}")
                return current_node

            queue.extend(current_node.children)

        logger.warning(f"No node found with task_id={task_id} and step_id={step_id}")
        return None

    async def select_step_and_agent(
        self,
        task: Dict,
    ):
        """Use MCTS to select the best step and agent for the task"""

        task_id = str(task["_id"])
        if not task_id:
            logger.error("Task ID is required to select a step and agent")
            return

        if task_id in self.root_nodes:
            root = self.root_nodes[task_id]
        else:
            # Create a new root node for this task
            root = TreeNode(
                task_id=task_id,
                step_id=str(ObjectId()),
                step_description=None,
            )
            self.root_nodes[task_id] = root

        # Initialize untried steps only if the root is brand new (has no untried steps yet)
        if not root.untried_steps and not root.children:
            initial_steps_response = await self._generate_initial_steps(
                task, self.capabilities_list
            )
            if not initial_steps_response:
                logger.error("Failed to generate initial steps.")
                return

            # -- Prevent duplicates from the start: (though root has no path, just in case)
            unique_steps = self.filter_out_duplicate_steps(
                initial_steps_response.estimated_steps, set()
            )
            # Limit the number of untried steps
            root.untried_steps = unique_steps[: self.max_children_per_node]
            logger.info(
                f"Initialized root node with {len(root.untried_steps)} untried steps"
            )

        # Run MCTS iterations
        for i in range(self.max_experiments):
            logger.info(f"Starting MCTS iteration {i+1}")
            node = await self._tree_policy(root)

            if node.step_description is None and node != root:
                logger.warning("Tree policy returned node with no step description")
                continue

            # Only simulate if we have a valid node and simulation hasn't been sent
            if node != root and node.step_description and not node.is_simulation_sent:
                # Find agent for selected step
                agent_id = await self._select_agent_for_step(
                    node.step_description, self.agent_details
                )
                if not agent_id:
                    logger.warning(
                        f"No suitable agent found for step '{node.step_description}'"
                    )
                    continue
                node.agent_id = agent_id

                # Send the selected step and agent to the Boss via queue
                await self._simulate(node, task)
                node.is_simulation_sent = True  # Mark as simulation sent

            # We do NOT backpropagate here because the step execution (and thus reward)
            # will come later from from_boss_results.

        # Update the task tree structure
        await self.update_task_tree_structure(task)

    async def _simulate(self, node: TreeNode, task: Dict):
        """Send the task step and agent to BOSS for execution"""
        logger.info(f"Simulating node {node} with state {node.step_description}")

        if node.step_description is None or node.agent_id is None:
            logger.warning(
                f"Cannot simulate node with no step description or agent for task {node.task_id}"
            )
            return

        final_targets = node.targets if node.targets else task.get("targets", [])

        # Prepare the task execution request
        execution_request = {
            "task_id": node.task_id,
            "step_id": node.step_id,
            "step_description": node.step_description,
            "agent_id": node.agent_id,
            "task_description": task.get("description", ""),
            "targets": final_targets,
        }

        # Send the request to BOSS via the queue
        await self.selfplay_response_queue.put(execution_request)

        # Store the node in pending simulations
        simulation_key = f"{node.task_id}_{node.step_id}"
        self.pending_simulations[simulation_key] = node

        # Since the task is asynchronous, we don't return a reward here

    async def update_task_in_database(self, task: Dict):
        task_id = task.get("_id")
        if not task_id:
            logger.error("Task does not have an '_id' field.")
            return

        # Ensure task_id is an ObjectId
        if isinstance(task_id, str):
            task_id = ObjectId(task_id)

        # Remove '_id' from the task to prevent attempting to update it
        task_without_id = {k: v for k, v in task.items() if k != "_id"}

        # Update the task in the database
        result = await self.tasks_collection.update_one(
            {"_id": task_id},
            {"$set": task_without_id},
            upsert=True,
        )

        # Logging the result
        task_id_str = str(task_id)
        if result.matched_count:
            logger.info(f"Successfully updated task {task_id_str}.")
        elif result.upserted_id:
            logger.info(f"Inserted new task with id {result.upserted_id}.")
        else:
            logger.warning(f"No changes made to task {task_id_str}.")

    async def _generate_initial_steps(
        self, task: Dict, capabilities_list: str
    ) -> Optional[StepGenerationResponse]:
        """
        Generate initial steps for a task based on its description.
        """
        try:
            task_description = task.get("description", "")

            if not task_description:
                logger.error(f"Task {task['_id']} has no description.")
                return None

            # Generate steps using the LLM
            # Pass empty path context for the root
            steps_response = await self._generate_steps_from_description(
                description=task_description,
                capabilities_list=self.capabilities_list,
                path_context="",
            )

            if not steps_response or not steps_response.estimated_steps:
                logger.error(f"No initial steps generated for task {task['_id']}.")
                return None

            return steps_response

        except Exception as e:
            logger.error(
                f"Error generating initial steps for task {task.get('_id', 'Unknown')}: {e}"
            )
            return None

    async def _generate_steps_from_description(
        self, description: str, capabilities_list: str, path_context: str
    ) -> Optional[StepGenerationResponse]:
        """
        Generate steps from task description, existing path context, and agent capabilities.
        """
        try:
            if not description:
                logger.error("Empty task description provided for step generation.")
                return None

            messages = BossPrompts.format_step_generation(
                task_description=description,
                capabilities_list=capabilities_list,
                path_history=path_context,
            )

            logger.debug(f"Generated messages for step generation: {messages}")

            estimation = await call_openai_api_structured(
                messages=messages,
                response_model=StepGenerationResponse,
                model="gpt-4o-mini",
            )

            if not estimation or not estimation.estimated_steps:
                logger.error("No steps generated by the LLM.")
                return None

            return estimation

        except Exception as e:
            logger.error(
                f"Error in generating steps from description: {e}", exc_info=True
            )
            return None

    async def _tree_policy(self, node: TreeNode, max_depth: int = 3) -> TreeNode:
        """Select a node to expand or explore asynchronously."""
        current_depth = 0
        current_node = node

        while not self._is_terminal(current_node) and current_depth < max_depth:
            if not current_node.fully_expanded():
                # Try to expand
                expanded_node = self._expand(current_node)
                if expanded_node != current_node:  # Successful expansion
                    # Wait for untried steps to be initialized (child might do it asynchronously)
                    while (
                        not expanded_node.untried_steps and not expanded_node.is_leaf()
                    ):
                        await asyncio.sleep(0.1)
                    logger.info(
                        f"Successfully expanded node with step: {expanded_node.step_description}"
                    )
                    return expanded_node
                # If expansion returns same node, continue with selection

            # Only try UCT select if we have children
            if current_node.children:
                current_node = self._uct_select(current_node)
                current_depth += 1
            else:
                # No expansion possible and no children - we're stuck
                logger.warning("No expansion possible and no children available")
                break

        return current_node

    def _expand(self, node: TreeNode) -> TreeNode:
        """Expand the node by trying an untried step."""
        if not node.untried_steps:
            logger.warning(f"No untried steps available for node {node.task_id}")
            return node  # Return same node to indicate expansion failure

        # Limit the number of children per node
        if len(node.children) >= self.max_children_per_node:
            logger.info(
                f"Node has reached maximum children ({self.max_children_per_node})"
            )
            return node  # Do not expand further

        step_found = None
        while node.untried_steps and not step_found:
            step = node.untried_steps.pop()
            step_description = step.step_description
            if not step_description:
                logger.error("Invalid step description encountered; skipping.")
                continue

            # 1) Ensure no duplicate step in the *current path*
            path_descriptions = set(self.get_path_to_node(node))
            if step_description in path_descriptions:
                logger.info(
                    f"Skipping duplicate step '{step_description}' in the current path."
                )
                continue

            # If it's truly new, use it
            step_found = step

        if not step_found:
            logger.info("No suitable new step found to expand.")
            return node  # No new step was found

        # Create new child node
        child_node = TreeNode(
            task_id=node.task_id,
            step_id=str(ObjectId()),
            step_description=step_found.step_description,
            targets=step_found.targets,
            parent=node,
        )
        node.add_child(child_node)

        # Initialize untried steps for the child node asynchronously
        asyncio.create_task(self._initialize_child_untried_steps(child_node))
        logger.info(
            f"Successfully created child node with step: {child_node.step_description}"
        )
        return child_node

    async def _initialize_child_untried_steps(self, child_node: TreeNode):
        """
        Initialize untried steps for the child node asynchronously,
        taking into account the path (including evaluations) so far.
        """
        task = self.get_task_by_id(child_node.task_id)
        if not task:
            logger.error(f"Task with id {child_node.task_id} not found.")
            return

        task_description = task.get("description", "")
        if not task_description:
            logger.error("Task description is empty.")
            return

        # Collect the entire path context, including evaluations
        path_context = self.collect_path_context(child_node)

        # Generate steps from the current description and path context
        steps_response = await self._generate_steps_from_description(
            description=task_description,
            capabilities_list=self.capabilities_list,
            path_context=path_context,
        )

        if not steps_response or not steps_response.estimated_steps:
            logger.warning("No steps generated for the child node.")
            return

        # Filter out duplicates and limit the number of untried steps
        existing_path = set(self.get_path_to_node(child_node))
        filtered_steps = self.filter_out_duplicate_steps(
            steps_response.estimated_steps, existing_path
        )
        child_node.untried_steps = filtered_steps[: self.max_children_per_node]

    def filter_out_duplicate_steps(
        self, steps: List[EstimatedNode], existing_path: set
    ) -> List[EstimatedNode]:
        """
        Remove any steps whose descriptions already exist in the 'existing_path' set.
        """
        unique = []
        for s in steps:
            if s.step_description not in existing_path:
                unique.append(s)
            else:
                logger.info(
                    f"Filtering out duplicate generated step: '{s.step_description}'"
                )
        return unique

    def _uct_select(self, node: TreeNode) -> TreeNode:
        """
        Select the best child node using the UCT formula.
        """
        log_parent_visits = math.log(node.visits + 1)

        uct_scores = {}
        for child in node.children:
            exploitation = child.value / max(child.visits, 1)
            exploration = self.exploration_constant * math.sqrt(
                log_parent_visits / max(child.visits, 1)
            )
            uct_score = exploitation + exploration

            uct_scores[child] = uct_score
            child.uct_score = uct_score  # Store for visualization

            step_description = child.step_description or "N/A"

            # Record exploration details
            exploration_detail = (
                f"Step: {step_description} "
                f"(Exploit: {exploitation:.3f}, "
                f"Explore: {exploration:.3f}, "
                f"Total: {uct_score:.3f})"
            )
            node.exploration_history.append(exploration_detail)

        if not uct_scores:
            logger.warning("No UCT scores available for selection")
            return node  # Return current node if no children to select

        selected_child = max(uct_scores.items(), key=lambda x: x[1])[0]

        selected_step_description = selected_child.step_description or "N/A"
        node.exploration_history.append(f"Selected Step: {selected_step_description}")
        return selected_child

    async def _select_agent_for_step(
        self, step_description: str, agent_details: List[Dict]
    ) -> Optional[str]:
        """Select the best agent for the given step."""
        agent_details_str = "\n".join(
            [
                f"Agent {agent['agent_id']}: {', '.join(agent.get('capabilities', []))}"
                for agent in agent_details
            ]
        )

        messages = BossPrompts.format_agent_selection(
            step_description=step_description,
            agent_details=agent_details_str,
        )

        response = await call_openai_api_structured(
            messages=messages,
            response_model=AgentSelectionAnalysis,
            model="gpt-4o-mini",
        )

        if not response:
            logger.error("Failed to select agent with LLM.")
            return None

        return response.agent_id

    async def _backpropagate(self, node: TreeNode, reward: float):
        """Backpropagate the reward through the tree and collect experiences."""
        while node is not None:
            node.update(reward)
            # Log evaluation details
            logger.info(f"Node evaluation: {node.llm_evaluation}")
            node = node.parent

    def _is_terminal(self, node: TreeNode) -> bool:
        """Check if node represents a terminal state."""
        # Add depth-based termination
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
            if depth > 20:  # Safeguard: large depth might indicate we should stop
                logger.warning("Reached maximum depth limit in MCTS tree.")
                return True

        # If no untried steps and no children, it is terminal
        if not node.untried_steps and not node.children:
            return True

        return False
