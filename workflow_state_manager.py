import logging
from typing import Dict, Optional

from bson import ObjectId

logger = logging.getLogger(__name__)


class WorkflowStateManager:
    def __init__(self, tasks_collection, task_history_collection, agents_collection):
        self.tasks_collection = tasks_collection
        self.task_history_collection = task_history_collection
        self.agents_collection = agents_collection

    def get_current_step(self, task_id: str) -> Optional[Dict]:
        task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})
        if not task:
            return None
        steps = task.get("steps", [])
        current_step_index = task.get("current_step_index")
        if current_step_index is not None and current_step_index < len(steps):
            return steps[current_step_index]
        else:
            return None

    def add_steps(self, task_id: str, additional_steps: list[Dict]):
        self.tasks_collection.update_one(
            {"_id": ObjectId(task_id)},
            {"$push": {"steps": {"$each": additional_steps}}},
        )
