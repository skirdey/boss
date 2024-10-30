import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from bson import ObjectId
from openai import OpenAI
from pydantic import BaseModel
from pymongo import ReturnDocument


class CapabilityEvaluation(BaseModel):
    can_handle: bool
    confidence: float
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

    def _get_object_id(self, id_value) -> Optional[ObjectId]:
        """Safely convert string to ObjectId"""
        if isinstance(id_value, ObjectId):
            return id_value
        try:
            return ObjectId(id_value)
        except Exception as e:
            self.logger.error(f"Invalid ObjectId: {id_value}. Error: {e}")
            return None

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
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=CapabilityEvaluation,
            )

            evaluation = completion.choices[0].message.parsed

            self.logger.debug(
                f"Capability evaluation: can_handle={evaluation.can_handle}, "
                f"confidence={evaluation.confidence}, reasoning={evaluation.reasoning}"
            )

            return evaluation.can_handle, evaluation.confidence

        except Exception as e:
            self.logger.error(f"Error evaluating agent capability: {e}")
            return False, 0.0

    def get_agent_load(self, agent_id: str) -> int:
        """Get current task load for an agent"""
        try:
            agent = self.agents_collection.find_one(
                {"agent_id": agent_id}, {"current_tasks": 1}
            )
            return agent.get("current_tasks", 0) if agent else 0
        except Exception as e:
            self.logger.error(f"Error getting agent load: {e}")
            return 0

    def update_agent_task_count(self, agent_id: str, increment: int = 1) -> bool:
        """Update agent's task count"""
        try:
            result = self.agents_collection.update_one(
                {"agent_id": agent_id},
                {
                    "$inc": {"current_tasks": increment},
                    "$set": {"last_updated": datetime.now(timezone.utc)},
                },
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating agent task count: {e}")
            return False

    def assign_task(self, task_id: str, task_description: str) -> Optional[str]:
        """
        Atomically find and assign a task to the most suitable agent.
        Returns the assigned agent_id if successful, else None.
        """
        obj_task_id = self._get_object_id(task_id)

        self.logger.debug(f"Assigning task {task_id} to an agent")

        if not obj_task_id:
            return None

        try:
            # Get all active agents with their current load
            active_agents = list(self.agents_collection.find({"status": "active"}))

            self.logger.debug(f"Active agents: {active_agents}")

            if not active_agents:
                self.logger.warning("No available agents found")
                return None

            # Evaluate each agent's capability
            capable_agents = []
            for agent in active_agents:
                can_handle, confidence = self.evaluate_agent_capability(
                    task_description, agent.get("capabilities", [])
                )

                self.logger.debug(
                    f"Agent {agent['agent_id']} can handle: {can_handle}, confidence: {confidence} for task: {task_description}"
                )

                if can_handle and confidence >= 0.0:  # Minimum confidence threshold
                    capable_agents.append(
                        {
                            "agent_id": agent["agent_id"],
                            "confidence": confidence,
                            "current_load": self.get_agent_load(agent["agent_id"]),
                        }
                    )

            if not capable_agents:
                self.logger.warning(f"No capable agents found for task {task_id}")
                return None

            # Sort by confidence and load
            capable_agents.sort(key=lambda x: (-x["confidence"], x["current_load"]))

            # Try to assign to each capable agent
            for agent in capable_agents:
                self.logger.debug(
                    f"Trying to assign task {task_id} to agent {agent['agent_id']}"
                )

                try:
                    # Atomically update task
                    updated_step = self.tasks_collection.find_one_and_update(
                        {
                            "_id": obj_task_id,
                            "workflow_state.remaining_steps.step_description": task_description,
                        },
                        {
                            "$set": {
                                "workflow_state.remaining_steps.$.agent_id": agent[
                                    "agent_id"
                                ],
                                "workflow_state.remaining_steps.$.assignment_time": datetime.now(
                                    timezone.utc
                                ),
                                "workflow_state.remaining_steps.$.assignment_metadata": {
                                    "confidence_score": agent["confidence"],
                                },
                            }
                        },
                        return_document=ReturnDocument.AFTER,
                    )

                    self.logger.debug(
                        f"Updated task: {updated_step} for agent {agent['agent_id']}"
                    )

                    return agent["agent_id"]

                except Exception as e:
                    self.logger.error(f"Error during task assignment: {e}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error in assign_task: {e}")
            return None
