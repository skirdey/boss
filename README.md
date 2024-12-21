# BOSS - Multi-Agent LLM Operating System For Offensive Security

BOSS is an intelligent task orchestration system that leverages Large Language Models (LLMs) to coordinate and execute agent-based workflows using Monte Carlo Tree Search (MCTS) and Self-Play for dynamic task planning and optimization.

## Core Features

- **Intelligent Task Decomposition:** Uses LLM-guided MCTS to break down complex tasks into optimal sequences of steps
- **Dynamic Agent Selection:** Employs LLM analysis to match agent capabilities with task requirements
- **Self-Play Optimization:** Continuously improves task planning through simulated execution and evaluation
- **Real-time Monitoring & Adaptation:** Tracks execution progress and optimizes workflows dynamically
- **Robust Error Handling:** Implements multiple retry strategies with intelligent failure analysis
- **Human-in-the-Loop:** Recognizes when to escalate tasks for human intervention
- **Performance Monitoring:** Continuously monitors system health and agent performance

> **Note:** This project is still under development and not all features are fully implemented. **Do not use in production.**

#### **This project currently focuses on network security related reasoning tasks, but BOSS can be extended to other domains with ease.**

## Architecture

```
+--------------------------------+
|       BOSS OPERATING SYS       |
+--------------------------------+
                |
                v
+--------------------------------+
|      Task Planning System       |
|   +----------------------------+|
|   |    Monte Carlo Tree Search ||
|   |    - Step Generation       ||
|   |    - Agent Selection       ||
|   |    - Path Optimization     ||
|   +----------------------------+|
|   |        Self-Play          ||
|   |    - Simulation           ||
|   |    - Evaluation           ||
|   |    - Experience Collection ||
|   +----------------------------+|
+--------------------------------+
                |
                v
+--------------------------------+
|     Message Bus (Kafka)        |
+--------------------------------+
                |
                v
+--------------------------------+
|        Agent Network           |
|    - Ping                      |
|    - WHOIS                     |
|    - SSL                       |
|    - REST Test                 |
|    - WebSocket Test            |
|    - Scan Ports                |
|    - Get SSL Cert              |
|    - API Explorer              |
|    - Conversation              |
|    - DIG Agent                 |
+--------------------------------+
                |
                v
+--------------------------------+
|      Result Processing         |
|    - Performance Evaluation    |
|    - State Updates            |
+--------------------------------+
                |
                v
+--------------------------------+
|          MongoDB               |
|    - Tasks                     |
|    - Agent Status             |
|    - Tree States              |
+--------------------------------+
```

## MCTS and Self-Play Implementation

### Monte Carlo Tree Search (MCTS)
BOSS uses an advanced implementation of MCTS for task planning:

- **Node Structure:** Each node represents a task state and contains:
  - Task/Step description
  - Agent assignment
  - Evaluation metrics
  - Visit counts and value estimates
  - Child nodes and unexplored actions

- **Tree Policy:**
  - Balances exploration and exploitation using UCT formula
  - Dynamically expands nodes based on LLM-generated steps
  - Limits tree depth and breadth for computational efficiency

- **Simulation:**
  - Uses actual agent execution results for state evaluation
  - Incorporates LLM-based performance assessment
  - Tracks simulation status to handle asynchronous execution

### Self-Play Integration

The self-play system:
- Simulates task execution with selected agents
- Evaluates performance using LLM-based criteria
- Collects experience data for optimization
- Updates tree statistics based on execution results

### LLM Integration

The LLM serves multiple roles:
- **Policy Network:** Generates possible next steps and actions
- **Value Network:** Evaluates state quality and action effectiveness
- **Agent Selection:** Analyzes agent capabilities for task matching
- **Performance Evaluation:** Assesses execution results and provides feedback

## Quick Start

### Local Setup

1. **Clone the Repository**

2. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Build Web Components:**
   ```bash
   cd web && docker compose build
   ```

4. **Start Infrastructure Services:**
   ```bash
   # In root directory
   docker compose up
   ```
   This starts:
   - Web UI
   - Kafka Message Broker
   - MongoDB Database

   Run `docker compose down -v && docker compose down && docker compose up` to clear kafka topics and volumes and restart the services.

5. **Run BOSS:**
   ```bash
   python boss/start.py
   ```

### Environment Variables

Required environment variables:
- **OPENAI_API_KEY:** API key for OpenAI services
- **MONGODB_URI:** MongoDB connection string
- **KAFKA_BOOTSTRAP_SERVERS:** Kafka broker address
- **ANTHROPIC_API_KEY:** API key for Anthropic services

## Agent Integration

To integrate new agents:

1. **Create Agent Class:**
```python
from boss.wrappers.wrapper_agent import WrapperAgent

class WrapperNewAgent(WrapperAgent):
    def process_task(self, task: Dict) -> Dict[str, Any]:
        result = {
            "task_id": task["_id"],
            "result": "Processed successfully",
            "metadata": {},
        }
        return result
```

2. **Register Agent:**
Add to `components_to_start` in `start.py`:
```python
components_to_start = [
    BOSS,
    WrapperPing,
    # ... other agents ...
    WrapperNewAgent,  # Your new agent
]
```

## Task States

```python
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
```

## References

This work builds upon and is inspired by several key papers in the field of LLM reasoning and optimization:

1. **LLaMA-Berry**: [LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning](https://arxiv.org/abs/2410.02884)
   - Introduces novel approaches for mathematical reasoning optimization in LLMs

2. **Marco-o1**: [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://arxiv.org/abs/2411.14405)
   - Paper: [arXiv](https://arxiv.org/abs/2411.14405)
   - Model: [Hugging Face - AIDC-AI/Marco-o1](https://huggingface.co/AIDC-AI/Marco-o1)

3. **LLaVA-o1**: [LLaVA-o1: Let Vision Language Models Reason Step-by-Step](https://arxiv.org/abs/2411.10440)
   - Demonstrates step-by-step reasoning capabilities in vision-language models

4. **LLaVA-CoT**: [Let Vision Language Models Reason Step-by-Step](https://github.com/PKU-YuanGroup/LLaVA-CoT)
   - Implementation of Chain-of-Thought reasoning for vision-language tasks

## Citation

If you use BOSS in your research or applications, please cite:

```bibtex
@software{boss2024,
  author = {Stanislav Kirdey},
  title = {BOSS: Multi-Agent LLM Operating System For Offensive Security},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/skirdey/boss}},
  commit = {253d93f48dfffe51fd7203f596f8ccdfd068fb96},
  note = {A multi-agent system leveraging LLMs for orchestrating offensive security tasks}
}

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.