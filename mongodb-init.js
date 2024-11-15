db = db.getSiblingDB('task_db');

// Tasks collection with enhanced schema
db.createCollection('tasks');

db.tasks.insertMany([
    
    {
        "_id": ObjectId(),
        "notes": "",
        "description": "find vulnerabilities on pieplatform.co",
        "status": "Created",

        "created_at": new Date(),
        "updated_at": new Date(),
        "evaluation_criteria": "Any information or entry points are discovered about the target",
        "current_step_index": null,
        "steps": [],
        "audit_trail": [],
    },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on developers.inflection.ai",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://pieplatform.co",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference` This endpoint generates a completion to a provided conversation. It requires the following headers, where <token> is the API credential you were provided by Inflection AI. Authorization: Bearer <token>",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference`",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },

    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on developers.inflection.ai",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://staging.pieplatform.co",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/docs",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference` This endpoint generates a completion to a provided conversation. It requires the following headers, where <token> is the API credential you were provided by Inflection AI. Authorization: Bearer <token>",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference`",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },

    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on developers.inflection.ai",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://pieplatform.co",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/` This endpoint generates a completion to a provided conversation. It requires the following headers, where <token> is the API credential you were provided by Inflection AI. Authorization: Bearer <token>",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/`",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },

    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on developers.inflection.ai",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://staging.pieplatform.co",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://www.inflection.ai/",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference` This endpoint generates a completion to a provided conversation. It requires the following headers, where <token> is the API credential you were provided by Inflection AI. Authorization: Bearer <token>",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference`",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },

    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on developers.inflection.ai",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://pieplatform.co",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground/",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground/_/",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference/` This endpoint generates a completion to a provided conversation. It requires the following headers, where <token> is the API credential you were provided by Inflection AI. Authorization: Bearer <token>",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference/_/`",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on webgoat sandbox apps located at 127.0.0.1:8080/WebGoat",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on developers.inflection.ai",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://pieplatform.com",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on https://developers.inflection.ai/playground",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference` This endpoint generates a completion to a provided conversation. It requires the following headers, where <token> is the API credential you were provided by Inflection AI. Authorization: Bearer <token>",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // },
    // {
    //     "_id": ObjectId(),
    //     "notes": "",
    //     "description": "find vulnerabilities on `https://layercake.pubwestus3.inf7ks8.com/external/api/inference`",
    //     "status": "Created",

    //     "created_at": new Date(),
    //     "updated_at": new Date(),
    //     "evaluation_criteria": "Any information or entry points are discovered about the target",
    //     "current_step_index": null,
    //     "steps": [],
    //     "audit_trail": [],
    // }
]);

// Agents collection with enhanced capabilities
db.createCollection('agents');
db.agents.insertMany([
    // {
    //     "agent_id": "agent_network_ping",
    //     "capabilities": [
    //         "can interact with command line interface to call `ping` command, this agent is able to ping a given IP or a domain and return the results without any additional setup in a single step"
    //     ],
    //     "status": "active",

    // },
    // {
    //     "agent_id": "agent_conversation",
    //     "capabilities": [
    //         "can analyze data, answer questions, and provide broad insights (do not have access to CLI or internet) - this agent is perfect for research, reasoning, data analysis, etc."
    //     ],
    //     "status": "active",

    // },
    // {
    //     "agent_id": "agent_network_scan_ports",
    //     "capabilities": [
    //         "can scan ports on a given IP or domain and return the results without any additional setup in a single step"
    //     ],
    //     "status": "active",
    // },
    // {
    //     "agent_id": "agent_network_get_ssl_certificate",
    //     "capabilities": [
    //         "can get the SSL certificate from a given domain, hostname, or IP address and return the results without any additional setup in a single step. This agent might also return no certificate information if resource doesn't employ SSL certificate."
    //     ],
    //     "status": "active",
    // },
    // {
    //     "agent_id": "agent_network_whois_lookup",
    //     "capabilities": [
    //         "can perform a whois lookup on a given domain, hostname, or IP address and return the results without any additional setup in a single step"
    //     ],
    //     "status": "active",
    // },
    // {
    //     "agent_id": "agent_network_dig_lookup",
    //     "capabilities": ["can perform a dig lookup on a given domain and return the results without any additional setup in a single step"],
    //     "status": "active",
    // },
    // {
    //     "agent_id": "agent_rest_tester",
    //     "capabilities": ["High priority agent. can test REST endpoints by sending requests and receiving responses"],
    //     "status": "active",
    // },
    // {
    //     "agent_id": "agent_websocket_tester",
    //     "capabilities": ["High priority agent. can test WebSocket endpoints by sending messages and receiving responses"],
    //     "status": "active",
    // },
    {
        "agent_id": "agent_api_explorer",
        "capabilities": ["High priority agent. comprehensive web security tool that performs tasks such as port scanning, SSL certificate analysis, subdomain enumeration, directory and file discovery, detection of web technologies, and verification of security headers. Additionally, it identifies potential vulnerabilities like CORS misconfigurations, CSRF weaknesses, injection points, and sensitive file disclosures, compiling all findings into a structured JSON report."],
        "status": "active",
    },
    {
        "agent_id": "agent_sql_injection_tester",
        "capabilities": ["High priority agent. This agent uses an LLM to generate and execute SQL injection tests against a target URL and specified paths. It analyzes responses for vulnerabilities, assessing their severity and reporting findings, but is designed for educational environments and controlled testing due to its reliance on predictable patterns and lack of robust input validation. Agent requires a list of paths and target urls to perform the tests"],
        "status": "active",
    },

    // {
    //     "agent_id": "agent_html_analyzer",
    //     "capabilities": ["High priority agent. This agent grabs HTML content and does a basic analysis of the fields like forms, text areas, etc. It returns the HTML raw content with its findings. Agent input expects a list of dictionaries with the target and port such as `{'target': 'https://example.com', 'port': 80}`"],
    //     "status": "active",
    // },
    
]);

// Task history collection for audit trail
db.createCollection('task_history');
db.task_history.createIndex({ "task_id": 1 });
db.task_history.createIndex({ "timestamp": 1 });
