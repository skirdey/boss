import logging
import os
from typing import List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel

from utils import get_object_id


class CapabilityEvaluation(BaseModel):
    can_handle: bool
    reasoning: str


class AgentScheduler:
    """
    Enhanced agent scheduler with improved capability matching and task status tracking.
    Uses structured output parsing with GPT-4O.
    """

    def __init__(self, tasks_collection, agents_collection):
        self.tasks_collection = tasks_collection
        self.agents_collection = agents_collection
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = logging.getLogger(__name__)

    def evaluate_agent_capability(
        self, task_description: str, agent_capabilities: List[str]
    ) -> Tuple[bool, float]:
        """
        Evaluate if an agent can handle a task with confidence score.
        Returns (can_handle, confidence_score)
        """

        system_prompt = """
        You are an AI capability evaluator. Analyze if an agent with the given capabilities 
        can handle the specified task. Provide a structured evaluation including whether 
        the agent can handle the task, a confidence score, and brief reasoning.
        """

        user_prompt = f"""
        Task Description: {task_description}
        
        Agent Capabilities:
        {', '.join(agent_capabilities)}
        """

        self.logger.debug(f"Evaluating capability match for task: {task_description}")
        self.logger.debug(f"Agent capabilities: {agent_capabilities}")

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=CapabilityEvaluation,
            )

            evaluation = completion.choices[0].message.parsed

            self.logger.debug(
                f"Capability evaluation: can_handle={evaluation.can_handle}, "
                f"reasoning={evaluation.reasoning}"
            )

            return evaluation.can_handle

        except Exception as e:
            self.logger.error(f"Error evaluating agent capability: {e}")
            return False

    def find_agent_for_step(self, task_id: str, task_description: str) -> Optional[str]:
        """
        Atomically find and assign a task to the most suitable agent.
        Returns the assigned agent_id if successful, else None.
        """
        obj_task_id = get_object_id(task_id)

        self.logger.debug(f"Assigning task {task_id} to an agent")

        if obj_task_id is None:
            self.logger.error(f"Invalid task ID: {task_id}")
            return None

        try:
            # Get all active agents with their current load
            active_agents = list(self.agents_collection.find({"status": "active"}))

            self.logger.debug(f"Active agents: {active_agents}")

            if not active_agents:
                self.logger.warning("No available agents found")
                return None

            # Evaluate each agent's capability

            for agent in active_agents:
                can_handle = self.evaluate_agent_capability(
                    task_description, agent.get("capabilities", [])
                )

                self.logger.debug(
                    f"Agent {agent['agent_id']} can handle: {can_handle}, for task: {task_description}"
                )

                if can_handle:
                    # TODO - assign agent to task

                    return agent["agent_id"]

            return None

        except Exception as e:
            self.logger.error(f"Error in assign_task: {e}")
            return None
