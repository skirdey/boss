db = db.getSiblingDB('task_db');

// Tasks collection with enhanced schema
db.createCollection('tasks');

db.tasks.insertMany([
    {
        "_id": ObjectId(),
        "notes": "",
        "description": "Ping 4.4.4.4 and 8.8.8.8",
        "status": "Created",

        "created_at": new Date(),
        "updated_at": new Date(),
        "evaluation_criteria": "Pings are completed and results are returned",
        "steps": [],
        "audit_trail": [],
    },
    {
        "_id": ObjectId(),
        "notes": "",
        "description": "Ping 8.8.8.8",
        "status": "Created",

        "created_at": new Date(),
        "updated_at": new Date(),
        "evaluation_criteria": "Ping is completed and results are returned",
        "steps": [],
        "audit_trail": [],
    },
    {
        "_id": ObjectId(),
        "notes": "",
        "description": "What is the reason of life",
        "status": "Created",

        "created_at": new Date(),
        "updated_at": new Date(),
        "evaluation_criteria": "A non-empty answer is returned",
        "current_step_index": null,
        "steps": [],
        "audit_trail": [],
    }
]);

// Agents collection with enhanced capabilities
db.createCollection('agents');
db.agents.insertMany([
    {
        "agent_id": "agent_network_ping",
        "capabilities": [
            "can interact with command line interface to call `ping` command, this agent is able to ping a given IP or a domain and return the results without any additional setup in a single step"
        ],
        "status": "active",

    },
    {
        "agent_id": "agent_conversation",
        "capabilities": [
            "can analyze data, answer questions, and provide broad insights (do not have access to CLI or internet) - this agent is perfect for research, reasoning, data analysis, etc."
        ],
        "status": "active",

    }
]);

// Task history collection for audit trail
db.createCollection('task_history');
db.task_history.createIndex({ "task_id": 1 });
db.task_history.createIndex({ "timestamp": 1 });
