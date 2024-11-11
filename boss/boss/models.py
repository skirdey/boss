from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NecessityCheckResponse(BaseModel):
    is_additional_steps_needed: str  # Expected values: 'Yes' or 'No'
    justification: str


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
    explanation: str = Field(..., description="Explanation of the evaluation decision")


class TaskComplexityResponse(BaseModel):
    complexity: str
    priority_score: float
    reasoning: str
    estimated_time_minutes: int


class StepEstimate(BaseModel):
    """Model for individual step estimates"""

    step_description: str = Field(
        ..., description="Description of what needs to be done"
    )
    estimated_duration_minutes: int = Field(
        ..., description="Estimated time in minutes"
    )
    confidence_score: float = Field(..., description="Confidence score between 0 and 1")
    expected_outcome: str = Field(..., description="Expected result of the step")
    assigned_agent: Optional[str] = Field(
        None, description="Suggested agent for this step"
    )
    is_critical: bool = Field(
        description="Whether this step is on the critical path - True or False"
    )

    requires_previous_step: bool = Field(
        description="Whether this step requires the previous step to be completed"
    )


class StepEstimationResponse(BaseModel):
    """Response model for step estimation"""

    estimated_steps: List[StepEstimate] = Field(
        default_factory=list, description="List of estimated steps"
    )
    critical_path_steps: List[str] = Field(
        default_factory=list,
        description="List of step descriptions that are on the critical path",
    )
    additional_steps: Optional[List[StepEstimate]] = Field(
        default=None, description="Optional additional steps that might be needed"
    )
    overall_plan: str = Field(description="Overall plan for the execution")


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
