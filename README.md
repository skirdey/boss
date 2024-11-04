# BOSS - LLM Operating System

BOSS is an intelligent task orchestration system that leverages Large Language Models (LLMs) to coordinate and execute agent-based workflows. Think of it as a smart task manager that can:

- **Break Down Complex Tasks:** Decompose intricate tasks into manageable, actionable steps.
- **Smart Agent Selection:** Assign the most suitable agent for each step based on capabilities.
- **Real-time Monitoring & Adaptation:** Track execution progress, handle failures, and optimize workflows on the fly.
- **Robust Error Handling:** Implement multiple retry strategies with intelligent failure analysis.
- **Human-in-the-Loop:** Recognize when to escalate tasks for human intervention.
- **Performance Monitoring:** Continuously monitor system health and agent performance.

> **Note:** This project is still under development and not all features are fully implemented. **Do not use in production.**
> *This project currently focuses on network security related reasoning tasks. At the same time,BOSS can be extended to other domains with ease.*

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Local Setup](#local-setup)
  - [Launching BOSS](#launching-boss)
  - [Integrating Agents](#integrating-agents)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Agent Integration](#agent-integration)
- [Task States](#task-states)
- [License](#license)

## Key Features

- **Intelligent Task Analysis:** Automatically assesses task complexity and required steps.
- **Smart Agent Selection:** Matches tasks with the most capable agents.
- **Real-time Adaptation:** Adjusts workflows based on performance and results.
- **Robust Error Handling:** Implements multiple retry strategies with intelligent failure analysis.
- **Human-in-the-Loop:** Recognizes when to request human intervention.
- **Performance Monitoring:** Tracks system health and agent performance.
- **Agent Wrappers:** Provides an abstract interface (`WrapperAgent`) for integrating various agents seamlessly.
- **Prompt Management:** Utilizes structured prompts (`BossPrompts`) to interact with LLMs for task planning and evaluation.

## Architecture

```plaintext
┌───────────────────────────────────────────────────┐
│                 BOSS Core System                   │
├───────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌──────────────┐  ┌──────────┐  │
│ │Task Analysis│  │Agent Selection│  │Monitoring│  │
│ └─────────────┘  └──────────────┘  └──────────┘  │
│ ┌─────────────┐  ┌──────────────┐  ┌──────────┐  │
│ │Retry Logic  │  │Error Handling│  │Adaptation │  │
│ └─────────────┘  └──────────────┘  └──────────┘  │
└─────────────────────┬─────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────┐
│                Message Bus                         │
│                  (Kafka)                           │
└─────────────────────┬─────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────┐
│               Agent Network                        │
├───────────────┬───────────────┬───────────────────┤
│    Agent 1    │    Agent 2    │     Agent N       │
│ (WrapperAgent │ (WrapperAgent │ (WrapperAgent     │
│    Subclass)  │    Subclass)  │    Subclass)      │
└───────────────┴───────────────┴───────────────────┘
```

### Components

- **BOSS Core System:** Manages task orchestration, agent selection, monitoring, and error handling.
- **Message Bus (Kafka):** Facilitates communication between BOSS and agents.
- **Agent Network:** Comprises various agents implemented as subclasses of `WrapperAgent`, each tailored to specific capabilities.

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
   This command starts:
   - **Web UI**
   - **Kafka Message Broker**
   - **MongoDB Database**
   - **Zookeeper** (required for Kafka)

### Launching BOSS

1. **Initialize the Orchestration System:**
   ```bash
   python ./start.py
   ```
   This script initializes BOSS and its agents, setting up necessary connections and listeners.

### Integrating Agents

1. **Create a New Agent:**
   - Subclass the `WrapperAgent` abstract class.
   - Implement the `process_task` method with your agent's specific logic.

   ```python
   # agents/my_agent.py

   from wrappers.wrapper_agent import WrapperAgent

   class MyAgent(WrapperAgent):
       def process_task(self, task: Dict) -> Dict[str, Any]:
           # Implement task processing logic
           result = {
               "task_id": task["_id"],
               "result": "Task completed successfully.",
               "status": "success",
               # Add additional fields as needed
           }
           return result

   if __name__ == "__main__":
       agent = MyAgent(agent_id="my_agent_id")
       agent.start()
   ```

2. **Run the Agent:**
   ```bash
   python start.py
   ```
   The agent will start listening for tasks assigned to it via Kafka and send back results upon completion.

## How It Works

1. **Task Submission**
   - Tasks are submitted to BOSS via Kafka.
   - BOSS analyzes the task using LLMs and determines the required steps and resources.

2. **Task Orchestration**
   - BOSS breaks down the task into manageable steps.
   - Each step is assigned to the most suitable agent based on capabilities.

3. **Agent Execution**
   - Agents receive tasks through Kafka.
   - Upon processing, agents send back results to BOSS.

4. **Monitoring & Adaptation**
   - BOSS monitors task progress and agent performance in real-time.
   - Handles failures with retry strategies or escalates to human intervention if necessary.

5. **Final Evaluation**
   - Once all steps are completed, BOSS performs a final evaluation to ensure task completion meets all criteria.

Result in UI:
![Network Ping Example](imgs/ping_agent.png)

## Configuration

BOSS uses environment variables for configuration. Create a `.env` file in the root directory with the following content:

```env
OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=mongodb://localhost:27017
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Environment Variables

- **OPENAI_API_KEY:** API key for accessing OpenAI's services.
- **MONGODB_URI:** Connection string for the MongoDB database.
- **KAFKA_BOOTSTRAP_SERVERS:** Address of the Kafka broker.

## Agent Integration

To integrate new agents into BOSS, follow these steps:

1. **Subclass `WrapperAgent`:**
   - Create a new Python file for your agent.
   - Subclass the `WrapperAgent` abstract class.
   - Implement the `process_task` method with your agent's specific logic.

2. **Implement `process_task`:**
   - This method receives a task dictionary.
   - Process the task as per your agent's capabilities.
   - Return a result dictionary containing task outcomes.

3. **Start the Agent:**
   - Instantiate your agent class with a unique `agent_id`.
   - Call the `start` method to begin listening for tasks.

### Example:

```python
# agents/example_agent.py

from wrappers.wrapper_agent import WrapperAgent
from typing import Dict, Any

class ExampleAgent(WrapperAgent):
    def process_task(self, task: Dict) -> Dict[str, Any]:
        # Example processing logic
        task_description = task.get("description", "No description provided.")
        # Simulate task processing
        result = {
            "task_id": task["_id"],
            "result": f"Processed task: {task_description}",
            "status": "success",
            "metadata": {"processed_at": datetime.now(timezone.utc).isoformat()},
        }
        return result

if __name__ == "__main__":
    agent = ExampleAgent(agent_id="example_agent")
    agent.start()
```

## Task States

BOSS manages tasks through various states to ensure efficient orchestration and tracking. Below are the possible states a task can be in:

```python
from enum import Enum

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
    FINAL_COMPLETION = "Final_Complition"
```

### State Transitions

1. **Created → In_Progress:** When a task is assigned to an agent.
2. **In_Progress → Completed_Step:** Upon successful completion of a step.
3. **In_Progress → Failed:** If a step fails after retries.
4. **Failed → Awaiting_Human:** If the system cannot automatically resolve the failure.
5. **Completed_Step → Completed_Workflow:** When all steps are successfully completed.
6. **Any State → Paused:** If the task is manually paused.
7. **Any State → Final_Complition:** After final evaluation of the task.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## Additional Notes

- **Logging:** BOSS and agents use Python's `logging` module for detailed logs. Customize logging levels and formats as needed.
- **Extensibility:** The modular design allows for easy integration of new agents and expansion of capabilities.
- **Error Handling:** Comprehensive error handling ensures that failures are managed gracefully, with opportunities for retries or human intervention.
