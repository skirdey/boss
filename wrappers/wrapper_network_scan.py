import threading
import json
import subprocess
import logging
from pymongo import MongoClient
from datetime import datetime, timezone
from wrappers.wrapper_agent import WrapperAgent
import shlex
import os
import requests
import nmap  # You'll need to install python-nmap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WrapperNetworkScan(WrapperAgent):
    def __init__(self, agent_id='agent_network_scan', db_uri='mongodb://localhost:27017/',
                 kafka_bootstrap_servers='localhost:9092'):
        super().__init__(agent_id, db_uri, kafka_bootstrap_servers)
        self.task_logger = logging.getLogger(f"{self.agent_id}_tasks"
        try:
            import shutil
            nmap_path = shutil.which('nmap')
            if not nmap_path:
                raise RuntimeError("nmap is not installed or not in PATH")
            logger.info(f"Found nmap at: {nmap_path}")
            self.nm = nmap.PortScanner()
        except Exception as e:
            logger.error(f"Failed to initialize nmap: {e}")
            raise RuntimeError(f"Cannot initialize network scan agent: {str(e)}")

        # Add debug logging
        logger.setLevel(logging.DEBUG)
        logger.info(f"Network scan agent initialized with ID: {self.agent_id}")

    def process_task(self, task):
        if self.self_evaluate(task):
            try:
                # Extract target and scan parameters from the task description
                scan_params = self.extract_scan_parameters(task["description"])
                if scan_params is None:
                    return {
                        "task_id": task["_id"],
                        "result": "No valid scan parameters found in task description",
                        "success": False,
                        "note": "Invalid parameters"
                    }

                target = scan_params['target']
                ports = scan_params.get('ports', '1-1000')
                scan_type = scan_params.get('scan_type', '-sT')  # Default to TCP connect scan

                # Validate target
                if not self.is_valid_target(target):
                    return {
                        "task_id": task["_id"],
                        "result": "Invalid target specified",
                        "success": False,
                        "note": "Invalid target"
                    }

                # Execute the scan
                logger.info(f"Starting scan of {target} with parameters: ports={ports}, type={scan_type}")
                scan_result = self.nm.scan(target, ports, arguments=scan_type)

                # Process and format results
                formatted_results = self.format_scan_results(scan_result)
                
                return {
                    "task_id": task["_id"],
                    "result": formatted_results,
                    "success": True,
                    "scan_summary": self.nm.scaninfo(),
                    "timestamp": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"Scan error: {str(e)}")
                return {
                    "task_id": task["_id"],
                    "error": str(e),
                    "success": False,
                    "note": "Exception occurred during scan execution"
                }
        else:
            return {
                "task_id": task["_id"],
                "result": "Task not suitable for this agent",
                "success": False,
                "note": "Agent capabilities do not match task type"
            }

    def extract_scan_parameters(self, description):
        prompt = f"""
Extract network scan parameters from the following task description.

Examples:

Description: Scan port 80 on host 10.10.10.1
Parameters: {{"target": "10.10.10.1", "ports": "80", "scan_type": "-sT"}}

Description: Full port scan on 192.168.1.1
Parameters: {{"target": "192.168.1.1", "ports": "1-65535", "scan_type": "-sT"}}

Task Description:
"{description}"

Format the output as JSON with keys 'target', 'ports' (optional), and 'scan_type' (optional)
"""
        try:
            assistant_reply = self.call_openai_api(prompt)
            data = json.loads(assistant_reply)
            return data
        except Exception as e:
            logger.error(f"Error extracting scan parameters: {e}")
            return None

    def is_valid_target(self, target):
        """Validates if the target is a valid IP address or domain"""
        import re
        ip_regex = r'^(\d{1,3}\.){3}\d{1,3}$'
        domain_regex = r'^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'
        return bool(re.match(ip_regex, target) or re.match(domain_regex, target))

    def format_scan_results(self, scan_result):
        """Format nmap scan results into a structured format"""
        formatted = {
            "scan_info": scan_result.get('nmap', {}).get('scaninfo', {}),
            "hosts": {}
        }

        for host in scan_result.get('scan', {}):
            host_info = scan_result['scan'][host]
            formatted["hosts"][host] = {
                "state": host_info.get('status', {}).get('state', 'unknown'),
                "ports": []
            }
            
            for port in host_info.get('tcp', {}):
                port_info = host_info['tcp'][port]
                formatted["hosts"][host]["ports"].append({
                    "port": port,
                    "state": port_info.get('state'),
                    "service": port_info.get('name'),
                    "version": port_info.get('version', ''),
                    "product": port_info.get('product', '')
                })

        return formatted

    def self_evaluate(self, task):
        """Determine if this agent can handle the task"""
        return task.get("type") == 'network_scan'

    def get_agent_capabilities(self):
        return ['port_scan', 'service_detection']