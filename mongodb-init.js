db = db.getSiblingDB('task_db');

// Tasks collection with enhanced schema
db.createCollection('tasks');

db.tasks.insertMany([

    {
        "_id": ObjectId(),
        "notes": "",
        "description": "Explore http://127.0.0.1:8080/WebGoat/ for vulnerabilities",
        "status": "Created",
        "created_at": new Date(),
        "updated_at": new Date(),
        "evaluation_criteria": "Any information or entry points are discovered about the target",
        "use_mcts": true,
        "tree_structure": {
            "children": []
        }

    },

]);

// Agents collection with enhanced capabilities
db.createCollection('agents');
db.agents.insertMany([

    // works as expected
    {
        "agent_id": "agent_network_ping",
        "capabilities": [
            "can interact with command line interface to call `ping` command, this agent is able to ping a given IP or a domain and return the results without any additional setup in a single step"
        ],
        "status": "active",
        "priority": 50
    },

    // works as expected
    {
        "agent_id": "agent_conversation",
        "capabilities": [
            "can analyze data, answer questions, and provide broad insights (do not have access to CLI or internet) - this agent is perfect for research, reasoning, data analysis, etc."
        ],
        "status": "active",
        "priority": 50
    },

    // works as expected
    {
        "agent_id": "agent_network_dig_lookup",
        "capabilities": ["can perform a dig lookup on a given domain and return the results without any additional setup in a single step"],
        "status": "active",
        "priority": 50
    },


    // works as expected
    {
        "agent_id": "agent_html_analyzer",
        "capabilities": [
            "High priority agent. This gets raw HTML content and returns a JSON object with the raw HTML and its url. This agent requires a list of targets consiting of urls to operate on. \
            The WrapperHTMLAnalyzerAgent is a comprehensive tool designed to identify and explore a wide range of potential security vulnerabilities in web pages, including missing CSRF protection, potential XSS risks due to insufficient input validation, sensitive information in HTML comments, missing security headers such as Content-Security-Policy and X-Frame-Options, mixed content on HTTPS sites, insecure form actions, inline JavaScript and CSS, deprecated HTML elements, outdated libraries, improper form and input field validation, sensitive data field security, inline event handlers, iframe security, exposed API endpoints, and insecure session management, while also extracting all discovered URLs for further analysis."
        ],
        "status": "active",
        "priority": 100
    },

    // works as expected
    {
        "agent_id": "agent_network_get_ssl_certificate",
        "capabilities": [
            "can get the SSL certificate from a given domain, hostname, or IP address and return the results without any additional setup in a single step. This agent might also return no certificate information if resource doesn't employ SSL certificate."
        ],
        "status": "active",
        "priority": 67
    },

    {
        "agent_id": "agent_api_explorer",
        "capabilities": ["High priority agent. comprehensive web security tool that performs tasks such as port scanning, SSL certificate analysis, subdomain enumeration, directory and file discovery, detection of web technologies, and verification of security headers. Additionally, it identifies potential vulnerabilities like CORS misconfigurations, CSRF weaknesses, injection points, and sensitive file disclosures, compiling all findings into a structured JSON report."],
        "status": "active",
        "priority": 100
    },
    // {
    //     "agent_id": "agent_network_scan_ports",
    //     "capabilities": [
    //         "can scan ports on a given IP or domain and return the results without any additional setup in a single step"
    //     ],
    //     "status": "active",
    //     "priority": 99
    // },

    // // {
    // //     "agent_id": "agent_network_whois_lookup",
    // //     "capabilities": [
    // //         "can perform a whois lookup on a given domain, hostname, or IP address and return the results without any additional setup in a single step"
    // //     ],
    // //     "status": "active",
    // // },

    // {
    //     "agent_id": "agent_rest_tester",
    //     "capabilities": ["High priority agent. Can test REST endpoints by sending requests and receiving responses"],
    //     "status": "active",
    //     "priority": 95
    // },
    // // {
    // //     "agent_id": "agent_websocket_tester",
    // //     "capabilities": ["High priority agent. can test WebSocket endpoints by sending messages and receiving responses"],
    // //     "status": "active",
    // //     "priority": 70
    // // },

    // {
    //     "agent_id": "agent_sql_injection_tester",
    //     "capabilities": ["High priority agent. This agent uses an LLM to generate and execute SQL injection tests against a target URL and specified paths. It analyzes responses for vulnerabilities, assessing their severity and reporting findings, but is designed for educational environments and controlled testing due to its reliance on predictable patterns and lack of robust input validation. Agent requires a list of paths and target urls to perform the tests"],
    //     "status": "active",
    //     "priority": 90
    // },


]);

// Task history collection for audit trail
db.createCollection('task_history');
db.task_history.createIndex({ "task_id": 1 });
db.task_history.createIndex({ "timestamp": 1 });
