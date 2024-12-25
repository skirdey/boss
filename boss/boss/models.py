from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentCapabilityMatch(BaseModel):
    """Model for evaluating how well an agent's capabilities match task requirements"""

    capability_description: str = Field(
        ..., description="Natural language description of the capability"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this capability matches or partially matches",
    )


class AgentSelectionAnalysis(BaseModel):
    """Model for analyzing agent suitability for a task"""

    agent_id: str = Field(..., description="ID of the agent being analyzed")
    overall_match_score: int = Field(
        description="Overall suitability score [-1, 0, 1] which is -1 = low, 0 = medium, 1 = high"
    )
    capability_matches: List[AgentCapabilityMatch] = Field(
        ..., description="Analysis of each capability match"
    )
    load_assessment: str = Field(
        ..., description="Assessment of agent's current workload"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Potential risks in selecting this agent"
    )
    recommendation: str = Field(
        ..., description="Recommendation on whether to select this agent"
    )


class TaskState(str, Enum):
    CREATED = "Created"
    IN_PROGRESS = "In_Progress"
    WAITING_FOR_EVALUATION = "Waiting_For_Evaluation"
    AWAITING_HUMAN = "Awaiting_Human"
    COMPLETED_STEP = "Completed_Step"
    COMPLETED_WORKFLOW = "Completed_Workflow"
    FAILED = "Failed"
    PENDING_NEXT_STEP = "Pending_Next_Step"
    PAUSED = "Paused"
    FINAL_COMPLETION = "Final_Completion"


class TaskEvaluationResponse(BaseModel):
    """Structured response for task completion evaluation"""

    success: bool = Field(description="Whether the task was completed successfully")

    confidence_score: float = Field(
        ..., description="Confidence score as a float between 0.0 and 1.0"
    )

    reasoning: str = Field(
        description="Detailed explanation of the evaluation decision"
    )
    additional_steps_needed: List[str] = Field(
        default_factory=list,
        description="List of additional steps required if task is incomplete",
    )
    explanation: str = Field(..., description="Explanation of the evaluation decision")
    critique: str = Field(
        ...,
        description="Critique of the step completion considering the expected outcome",
    )

    agent_output: Optional[str]


class ScanFinding(BaseModel):
    """Model for individual scan findings"""

    parameter: str
    injection_type: str
    details: Dict[str, Any]
    timestamp: str
    severity: str