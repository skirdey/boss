# BOSS (Brain Operating System)

BOSS is an intelligent task orchestration system that uses LLMs (Large Language Models) to coordinate and execute agent-based workflows. Think of it as a smart task manager that can:
- Break down complex tasks into manageable steps
- Choose the best agent for each step
- Monitor execution and handle failures
- Adapt and optimize in real-time
- Know when to ask for human help

`
! Note - this project is still under development and not all features are fully implemented. Do not use in production.
`

## Key Features

- **Intelligent Task Analysis**: Automatically assesses task complexity and required steps
- **Smart Agent Selection**: Matches tasks with the most capable agents
- **Real-time Adaptation**: Adjusts workflows based on performance and results
- **Robust Error Handling**: Multiple retry strategies with intelligent failure analysis
- **Human-in-the-Loop**: Knows when to request human intervention
- **Performance Monitoring**: Tracks system health and agent performance

## Quick Start

### Local Setup

1. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Build web components:
```bash
cd web && docker compose build
```

3. Start infrastructure services:
```bash
# In root directory
docker compose up
```
This starts:
- Web UI
- Kafka message broker
- MongoDB database
- Zookeeper (required for Kafka)

4. Launch BOSS:
```bash
./start.py
```
This initializes the orchestration system and agents.

## Architecture

```
┌───────────────────────────────────────────────────┐
│             BOSS Core System                      │
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
│                  (Kafka)                          │
└─────────────────────┬─────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────┐
│              Agent Network                         │
├───────────────┬───────────────┬───────────────────┤
│    Agent 1    │    Agent 2    │     Agent N       │
│  (Capability  │  (Capability  │   (Capability     │
│    Set A)     │    Set B)     │     Set N)        │
└───────────────┴───────────────┴───────────────────┘
        │               │               │
        └───────────────▼───────────────┘
┌───────────────────────────────────────────────────┐
│                  Storage                          │
├─────────────────┬─────────────────┬──────────────┤
│     Tasks       │     History     │    Metrics    │
│   (MongoDB)     │    (MongoDB)    │   (MongoDB)   │
└─────────────────┴─────────────────┴──────────────┘
```

## How It Works

1. **Task Submission**
   - Tasks are submitted to MongoDB
   - BOSS analyzes complexity using LLMs
   - System estimates required steps and resources

2. **Task Orchestration**
   - BOSS coordinates task execution across agents
   - Real-time monitoring of progress and performance
   - Intelligent handling of failures and retries

3. **Agent Selection**
   - Matches task requirements with agent capabilities
   - Considers agent load and performance history
   - Dynamically adjusts assignments based on results

4. **Error Handling**
   - Multiple retry strategies with backoff
   - Intelligent failure analysis
   - Automatic escalation to human intervention when needed

## Example Usage

Task: Network connectivity check
```python
task = {
    "description": "ping 8.8.8.8",
    "priority": "medium",
    "timeout": 30
}
```

Result in UI:
![Network Ping Example](imgs/ping_agent.png)

## Configuration

BOSS uses environment variables for configuration. Create a `.env` file with:

```env
OPENAI_API_KEY=your_api_key
MONGODB_URI=mongodb://localhost:27017
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

## Advanced Features

- **Chain-of-Thought Reasoning**: Uses structured reasoning for complex decisions
- **Risk Assessment**: Evaluates potential issues before task execution
- **Performance Optimization**: Adapts check intervals based on task complexity
- **Parallel Execution**: Identifies tasks that can run simultaneously
- **Audit Trail**: Comprehensive history of all actions and decisions

## Task States

- `CREATED`: Initial task state
- `IN_PROGRESS`: Task is being executed
- `WAITING_FOR_EVALUATION`: Completed, pending success verification
- `AWAITING_HUMAN`: Requires human intervention
- `COMPLETED`: Successfully finished
- `FAILED`: Failed and exceeded retry attempts
- `PENDING_NEXT_STEP`: Ready for next step execution
- `PAUSED`: Temporarily suspended

## Contributing

Contributions welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and development process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.