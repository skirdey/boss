db = db.getSiblingDB('task_db');

// Tasks collection with enhanced schema
db.createCollection('tasks');
// Tasks collection with added step estimation fields
db.createCollection('tasks');
db.tasks.insertMany([
    {
        "_id": ObjectId(),
        "notes": "",
        "description": "Ping 8.8.8.8",
        "status": "Created",
        "retry_count": 0,
        "max_retries": 3,
        "is_deleted": false,
        "created_at": new Date(),
        "updated_at": new Date(),
        "estimated_total_steps": 1,
        "current_step": 0,
        "evaluation_criteria": "Ping is completed and results are returned",
        "workflow_state": {
            "completed_steps": [],
            "remaining_steps": [],
            "current_agent": null
        },
        // New fields for step estimation
        "step_estimation": {},
    },
    {
        "_id": ObjectId(),
        "notes": "",
        "description": "What is the reason of life",
        "status": "Created",
        "retry_count": 0,
        "max_retries": 3,
        "is_deleted": false,
        "created_at": new Date(),
        "updated_at": new Date(),
        "estimated_total_steps": 1,
        "evaluation_criteria": "An non-empty answer is returned",
        "current_step": 0,
        "workflow_state": {
            "completed_steps": [],
            "remaining_steps": [],
            "current_agent": null
        },
        // New fields for step estimation
        "step_estimation": {},
    }
]);

// Agents collection with enhanced capabilities
db.createCollection('agents');
db.agents.insertMany([
    {
        "agent_id": "agent_network_ping",
        "capabilities": [
            "can interact with command line interface to call `ping` command",
        ],
        "status": "active",
        "active_tasks": [
        ],

    },
    {
        "agent_id": "agent_conversation",
        "capabilities": [
            "can analyze data, answer questions and provide broad insights (do not have access to cli or internet)",

        ],
        "status": "active",
        "active_tasks": [
        ]
    },

]);



// Task history collection for audit trail
db.createCollection('task_history');
db.task_history.createIndex({ "task_id": 1 });
db.task_history.createIndex({ "timestamp": 1 });