db = db.getSiblingDB('task_db');

db.createCollection('tasks');
db.tasks.insertMany([
    {
        "_id": ObjectId(),
        "description": "Scan port 80 on host 10.10.10.1",
        "type": "network_scan",
        "status": "Created",
        "retry_count": 0,
        "max_retries": 3,
        "is_deleted": false,
        "created_at": new Date(),
        "updated_at": new Date()
    },
    {
        "_id": ObjectId(),
        "description": "Search for XYZ on GitHub",
        "type": "web_search",
        "status": "Created",
        "retry_count": 0,
        "max_retries": 3,
        "is_deleted": false,
        "created_at": new Date(),
        "updated_at": new Date()
    },
    {
        "_id": ObjectId(),
        "description": "ping 8.8.8.8",
        "type": "network_ping",
        "agent_id": "agent_network_ping",
        "status": "Created",
        "retry_count": 0,
        "max_retries": 3,
        "is_deleted": false,
        "created_at": new Date(),
        "updated_at": new Date()
    }
]); // Removed trailing comma

db.createCollection('agents');
db.agents.insertMany([
    {
        "agent_id": "agent_network_ping",
        "capabilities": ["ping"],
        "status": "active"
    } // Removed trailing comma
]);

db.createCollection('conv_template');
db.conv_template.insertMany([
    {
        "template_id": "template1",
        "content": "Hello, how can I assist you today?"
    }
]);
