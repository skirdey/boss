from typing import Dict, List


class BossPrompts:
    """
    Collection of prompts for the BOSS agent to handle task planning and delegation
    """

    # System prompt for step generation
    STEP_GENERATION_SYSTEM = """You are offensive security and penetration testing expert specialized in attack chain development."""

    # Step generation user prompt template
    STEP_GENERATION_USER = """

You are an AI assistant that helps generate actionable steps for tasks. Based on the original task and the current path, generate alternative next steps that can be performed by the available agents, considering their capabilities.

Original Task: {task_description}
Available Agents and their Capabilities:
{capabilities_list}

Task history:
{path_history}

Generate a list of possible next steps that:

- Are achievable using the capabilities of the available agents.
- Directly contribute to accomplishing the original task.
- Do not require capabilities that no agent possesses.

"""

    # System prompt for step evaluation
    STEP_EVALUATION_SYSTEM = """You are offensive security and penetration testing expert specialized in attack chain evaluation"""

    # Step evaluation user prompt template
    STEP_EVALUATION_USER = """Step Description:
{step_description}

Step Result:
{step_result}

Expected Outcome:
{expected_outcome}

Evaluate the step completion considering:
1. What new attack opportunities were revealed?
2. How does this affect operational security?
3. What advantages does this create for subsequent steps?
4. Should we adjust the attack chain based on these results?

Provide a structured evaluation including:
- Success/failure determination
- New intelligence gathered
- Attack chain implications
- Operational security assessment
- Recommended attack chain adjustments"""

    # System prompt for agent selection
    AGENT_SELECTION_SYSTEM = """You are offensive security and penetration testing expert specialized in agent orchestration. """

    # Agent selection user prompt template
    AGENT_SELECTION_USER = """Task Step:
{step_description}

Available Agents:
{agent_details}

Analyze agent suitability considering:
1. How well can the agent leverage current attack position
2. What unique capabilities can this agent provide for future steps
3. How does this agent's operation affect detection risk
4. Can this agent maintain existing access levels

Provide detailed analysis in AgentSelectionAnalysis format including:
- Capability synergy assessment
- Operational security impact
- Attack chain contribution
- Risk/benefit evaluation
- Final recommendation"""

    FINAL_EVALUATION_SYSTEM = """You are offensive security and penetration testing expert and You are an expert system for performing final task completion evaluation. 
"""

    # Final evaluation user prompt template
    FINAL_EVALUATION_USER = """Task Description:
{task_description}

Evaluation Criteria:
{step_description}

Agent Output:
{agent_output}

Agent ID:
{agent_id}

Evaluate overall task completion considering:
1. Most of step descriptioncriteria met
2. Logical consistency between steps


Provide evaluation in TaskEvaluationResponse format including:
- Success determination
- Confidence score
- Detailed reasoning
- Any additional steps needed
"""

    # All other methods remain exactly the same as in the original
    @staticmethod
    def format_additional_steps_necessity_check(
        task_description: str,
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are offensive security and penetration testing expert and You are an expert project manager helping to determine if additional steps are necessary to complete a task.",
            },
            {
                "role": "user",
                "content": (
                    f"{task_description}\n\n"
                    "Based on the above, do we truly need additional steps to complete this task? Please answer 'Yes' or 'No' and provide a brief justification."
                ),
            },
        ]

    @staticmethod
    def format_step_generation(
        task_description: str, capabilities_list: str, path_history: str
    ) -> dict:
        """Format the step generation prompts ensuring proper structure"""
        return [
            {"role": "system", "content": BossPrompts.STEP_GENERATION_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.STEP_GENERATION_USER.format(
                    task_description=task_description,
                    capabilities_list=capabilities_list,
                    path_history=path_history,
                ),
            },
        ]

    @staticmethod
    def format_step_evaluation(
        step_description: str, step_result: str, expected_outcome: str
    ) -> dict:
        """Format the step evaluation prompts"""
        return [
            {"role": "system", "content": BossPrompts.STEP_EVALUATION_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.STEP_EVALUATION_USER.format(
                    step_description=step_description,
                    step_result=step_result,
                    expected_outcome=expected_outcome,
                ),
            },
        ]

    @staticmethod
    def format_agent_selection(step_description: str, agent_details: str) -> dict:
        """Format the agent selection prompts"""
        return [
            {"role": "system", "content": BossPrompts.AGENT_SELECTION_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.AGENT_SELECTION_USER.format(
                    step_description=step_description, agent_details=agent_details
                ),
            },
        ]

    @staticmethod
    def format_final_evaluation(
        task_description: str, step_description: str, agent_output: str, agent_id: str
    ) -> dict:
        """Format the final evaluation prompts"""
        return [
            {"role": "system", "content": BossPrompts.FINAL_EVALUATION_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.FINAL_EVALUATION_USER.format(
                    task_description=task_description,
                    step_description=step_description,
                    agent_output=agent_output,
                    agent_id=agent_id,
                ),
            },
        ]

    @staticmethod
    def validate_prompt_format(messages: List[Dict[str, str]]) -> bool:
        """Validate that messages are properly formatted for OpenAI API"""
        if not isinstance(messages, list):
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["system", "user", "assistant"]:
                return False

        return True
