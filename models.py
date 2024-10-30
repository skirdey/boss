from dataclasses import dataclass
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


@dataclass
class StepResult:
    step_description: str
    result: Any
    success: bool
    execution_time: float
    error: Optional[str] = None
    metadata: Optional[Dict] = None

    @property
    def description(self) -> str:
        return self.step_description


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


class TaskComplexityResponse(BaseModel):
    complexity: str
    priority_score: float
    reasoning: str
    estimated_time_minutes: int


class StepEstimate(BaseModel):
    step_description: str
    estimated_duration_minutes: int
    confidence_score: float = Field(
        ..., description="Confidence score as a float between 0.0 and 1.0"
    )
    expected_outcome: str


class StepEstimationResponse(BaseModel):
    estimated_steps: List[StepEstimate]
    critical_path_steps: List[str]
    additional_steps: Optional[List[StepEstimate]]


@dataclass
class TaskComplexity:
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TaskPriority:
    score: float
    complexity: str
    reasoning: str
    estimated_time: int
