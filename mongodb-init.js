db = db.getSiblingDB('task_db');

// Tasks collection with enhanced schema
db.createCollection('tasks');
// Tasks collection with added step estimation fields
db.createCollection('tasks');
db.tasks.insertMany([
    {
        "_id": ObjectId(),
        "description": "Find all info about 8.8.8.8",
        "type": "comprehensive_scan",
        "status": "Created",
        "retry_count": 0,
        "max_retries": 3,
        "is_deleted": false,
        "created_at": new Date(),
        "updated_at": new Date(),
        "estimated_total_steps": 3,
        "current_step": 0,
        "workflow_state": {
            "completed_steps": [],
            "remaining_steps": [],
            "current_agent": null
        },
        // New fields for step estimation
        "step_estimation": {
            "estimated_steps": [
                {
                    "step_type": "ping",
                    "estimated_duration_minutes": 1,
                    "confidence_score": 0.95,
                    "expected_outcome": "Basic connectivity check",
                    "potential_findings": ["latency", "packet_loss"]
                },
                {
                    "step_type": "port_scan",
                    "estimated_duration_minutes": 5,
                    "confidence_score": 0.85,
                    "expected_outcome": "Open port discovery",
                    "potential_findings": ["open_ports", "service_versions"]
                },
                {
                    "step_type": "kali_cli",
                    "estimated_duration_minutes": 15,
                    "confidence_score": 0.75,
                    "expected_outcome": "Detailed enumeration",
                    "potential_findings": ["vulnerabilities", "misconfigurations"]
                }
            ],
            "total_estimated_duration": 21,
            "estimation_timestamp": new Date(),
            "estimation_model_version": "gpt-4",
            "estimation_confidence": 0.85
        },
        "evaluation_criteria": "Gather comprehensive information about the target IP",
        "is_continuous": false,
        "audit_trail": []
    }
]);

// Agents collection with enhanced capabilities
db.createCollection('agents');
db.agents.insertMany([
    {
        "agent_id": "agent_network_ping",
        "capabilities": ["ping", "basic_network_discovery"],
        "status": "active",
        "priority": 1
    },
    {
        "agent_id": "agent_network_scan",
        "capabilities": ["port_scan", "service_detection"],
        "status": "active",
        "priority": 2
    },
    {
        "agent_id": "agent_kali_cli",
        "capabilities": ["vulnerability_scan", "advanced_enumeration", "exploitation"],
        "status": "active",
        "priority": 3
    }
]);



// Task history collection for audit trail
db.createCollection('task_history');
db.task_history.createIndex({ "task_id": 1 });
db.task_history.createIndex({ "timestamp": 1 });