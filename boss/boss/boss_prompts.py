from typing import Dict, List


class BossPrompts:
    """
    Collection of prompts for the BOSS agent to handle task planning and delegation
    """

    # System prompt for step generation
    STEP_GENERATION_SYSTEM = """You are an expert planning assistant responsible for breaking down complex tasks into clear, actionable steps for execution by specialized AI agents. Your focus is on:

1. Chain-of-thought reasoning to break down tasks logically
2. Identifying natural breakpoints between different capabilities
3. Ensuring steps are atomic and independently verifiable
4. Considering available agent capabilities when planning
5. Organize steps in order of growing benefits - start with steps that will yield most useful information for future steps
6. In certain situations expected outcome can be just execution of the step, like `ping` command that can not reach the target, but it can still be useful for future steps

When generating steps, prioritize:
- Clarity and precision in step descriptions
- Logical sequencing and dependencies
- Appropriate granularity - not too broad or too narrow
- Measurable outcomes for verification
"""

    # Step generation user prompt template
    STEP_GENERATION_USER = """Task Description: {task_description}

Available Agents and Their Capabilities:
{capabilities_list}

Generate a structured sequence of steps following these rules:
1. Each step should be atomic and independently executable
2. Include clear success criteria for each step
3. Consider dependencies between steps
4. Match steps to appropriate agent capabilities
5. Maintain appropriate scope - neither too broad nor too narrow

Respond in JSON format matching the StepEstimationResponse model with:
- Detailed step descriptions
- Estimated duration for each step
- Confidence scores
- Expected outcomes
- Critical path identification
"""

    # System prompt for step evaluation
    STEP_EVALUATION_SYSTEM = """You are an expert quality assurance system responsible for evaluating task execution results against requirements. Your role is to:

1. Analyze step completion against stated objectives
2. Verify logical consistency of results
3. Identify any gaps or issues
4. Recommend corrections or additional steps if needed
5. Assess result quality and completeness
6. If the step has no specific result, like `ping` command that can not reach the target, but it can still be useful for future steps, return success as True

Focus on objective evaluation using:
- Explicit success criteria
- Result verification
- Logical analysis
- Chain-of-thought reasoning
"""

    # Step evaluation user prompt template
    STEP_EVALUATION_USER = """Step Description:
{step_description}

Step Result:
{step_result}

Evaluate the step completion using these criteria:
1. Does the result adress at least some of the step requirements?


Provide a structured evaluation considering:
- Success/failure determination
- Confidence score (0.0-1.0)
- Detailed reasoning
- Any additional steps needed
"""

    # System prompt for agent selection
    AGENT_SELECTION_SYSTEM = """You are an expert system for matching tasks to the most appropriate AI agents based on capabilities, workload, and risk factors. Your focus is on:

1. Analyzing task requirements against agent capabilities  
2. Evaluating agent availability and current load
3. Assessing execution risks and mitigation strategies
4. Making optimal agent selection recommendations

Consider these factors in your analysis:
- Capability alignment 
- Load balancing
- Risk management
- Overall suitability
"""

    # Agent selection user prompt template
    AGENT_SELECTION_USER = """Task Step:
{step_description}

Available Agents:
{agent_details}

Analyze agent suitability considering:
1. Capability match for the specific step
2. Current agent workload and availability 
3. Potential risks and mitigation needs
4. Overall recommendation

Provide detailed analysis in AgentSelectionAnalysis format including:
- Match scores
- Capability assessment
- Load evaluation  
- Risk factors
- Final recommendation
"""

    # System prompt for task complexity analysis
    TASK_COMPLEXITY_SYSTEM = """You are an expert system for analyzing task complexity and priority. Your role is to:

1. Assess overall task complexity
2. Determine appropriate priority level
3. Estimate resource requirements
4. Identify critical factors affecting execution

Consider these aspects:
- Technical complexity
- Dependencies and constraints
- Resource requirements
- Time sensitivity
"""

    # Task complexity user prompt template
    TASK_COMPLEXITY_USER = """Task Description:
{task_description}

Analyze the task considering:
1. Overall complexity level
2. Priority factors
3. Resource requirements
4. Time constraints

Provide analysis in TaskComplexityResponse format with:
- Complexity classification
- Priority score
- Detailed reasoning
- Time estimates
"""

    # Final evaluation system prompt
    FINAL_EVALUATION_SYSTEM = """You are an expert system for performing final task completion evaluation. Your role is to:

1. Verify all success criteria have been met
2. Ensure logical consistency across steps
3. Identify any remaining gaps
4. Validate overall task completion quality

Focus on comprehensive evaluation using:
- Success criteria validation
- Cross-step consistency
- Quality assessment
- Gap analysis
"""

    # Final evaluation user prompt template
    FINAL_EVALUATION_USER = """Task Description:
{task_description}

Evaluation Criteria:
{evaluation_criteria}

Step Results:
{step_results}

Evaluate overall task completion considering:
1. All success criteria met
2. Logical consistency between steps
3. Quality of deliverables
4. Any remaining gaps

Provide evaluation in TaskEvaluationResponse format including:
- Success determination
- Confidence score
- Detailed reasoning
- Any additional steps needed
"""

    @staticmethod
    def format_step_generation(task_description: str, capabilities_list: str) -> dict:
        """Format the step generation prompts ensuring proper structure"""
        return [
            {"role": "system", "content": BossPrompts.STEP_GENERATION_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.STEP_GENERATION_USER.format(
                    task_description=task_description,
                    capabilities_list=capabilities_list,
                ),
            },
        ]

    @staticmethod
    def format_step_evaluation(step_description: str, step_result: str) -> dict:
        """Format the step evaluation prompts"""
        return [
            {"role": "system", "content": BossPrompts.STEP_EVALUATION_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.STEP_EVALUATION_USER.format(
                    step_description=step_description, step_result=step_result
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
    def format_task_complexity(task_description: str) -> dict:
        """Format the task complexity prompts"""
        return [
            {"role": "system", "content": BossPrompts.TASK_COMPLEXITY_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.TASK_COMPLEXITY_USER.format(
                    task_description=task_description
                ),
            },
        ]

    @staticmethod
    def format_final_evaluation(
        task_description: str, evaluation_criteria: str, step_results: str
    ) -> dict:
        """Format the final evaluation prompts"""
        return [
            {"role": "system", "content": BossPrompts.FINAL_EVALUATION_SYSTEM},
            {
                "role": "user",
                "content": BossPrompts.FINAL_EVALUATION_USER.format(
                    task_description=task_description,
                    evaluation_criteria=evaluation_criteria,
                    step_results=step_results,
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
