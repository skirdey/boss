import threading
import json
import subprocess
import logging
from pymongo import MongoClient
from datetime import datetime, timezone
from wrappers.wrapper_agent import WrapperAgent
import shlex
import logging
import os
import requests


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WrapperPing(WrapperAgent):
    def __init__(self, agent_id='agent_network_ping', db_uri='mongodb://localhost:27017/',
                 kafka_bootstrap_servers='localhost:9092'):
        super().__init__(agent_id, db_uri, kafka_bootstrap_servers)

    def process_task(self, task):
        if self.self_evaluate(task):
            try:
                # Extract the command and arguments from the task description
                command, args = self.extract_command(task["description"])
                if command is None:
                    return {
                        "task_id": task["_id"],
                        "result": "No valid command found in task description",
                        "success": False,
                        "note": "Invalid command"
                    }

                # Log the command execution
                self.task_logger.info(f"Executing command: {command} {' '.join(args)}")

                # Execute the command using subprocess
                result = subprocess.run(
                    [command] + args,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                # Check for errors
                if result.returncode == 0:
                    # Success: Return the stdout
                    output = result.stdout.strip()
                    # Evaluate the output based on evaluation criteria, if any
                    if self.evaluate_response(output, task.get("evaluation_criteria", "")):
                        return {
                            "task_id": task["_id"],
                            "result": output,
                            "success": True
                        }
                    else:
                        return {
                            "task_id": task["_id"],
                            "result": output,
                            "success": False,
                            "note": "Evaluation criteria not met"
                        }
                else:
                    # Failure: Return the stderr
                    error_output = result.stderr.strip()
                    return {
                        "task_id": task["_id"],
                        "result": error_output,
                        "success": False,
                        "note": f"Command exited with return code {result.returncode}"
                    }

            except subprocess.TimeoutExpired:
                return {
                    "task_id": task["_id"],
                    "result": "Command timed out",
                    "success": False,
                    "note": "TimeoutExpired"
                }
            except Exception as e:
                return {
                    "task_id": task["_id"],
                    "error": str(e),
                    "success": False,
                    "note": "Exception occurred during command execution"
                }
        else:
            # Reject the task as not suitable
            return {
                "task_id": task["_id"],
                "result": "Task not suitable for this agent",
                "success": False,
                "note": "Agent capabilities do not match task type"
            }

    def extract_command(self, description):
        prompt = f"""
Extract the command from the following task description for os.system() call in python.

Examples: 

Description: ping 8.8.8.8
Command: ping 8.8.8.8

Description: ping google.com
Command: ping google.com

Description: ping asjkahdakjsdh
Command: ""

Task Description:
"{description}"

Format the output as JSON with keys 'command'
"""

        assistant_reply = self.call_openai_api(prompt)

        logger.info(f"Assistant reply for extract_command: {assistant_reply}")

        try:
            data = json.loads(assistant_reply)
            cmd_str = data.get('command')
            if not cmd_str:
                return None, None
            cmd_str = cmd_str.strip()
            cmd, args = self.validate_command(cmd_str)
            return cmd, args
        except json.JSONDecodeError:
            return None, None

    def validate_command(self, command):
        allowed_commands = ['ping']
        tokens = shlex.split(command)

        logger.info(f"Command Tokens: {tokens}")

        if not tokens:
            return None, None

        cmd = tokens[0]
        args = tokens[1:]

        if cmd not in allowed_commands:
            return None, None

        # Validate arguments
        args = self.validate_arguments(cmd, args)
        if args is None:
            return None, None

        return cmd, args

    def validate_arguments(self, cmd_name, args):
        """
        Validates command arguments to prevent misuse.
        For example, allows only certain domains or IP addresses.
        """
        logger.info(f"Validating arguments for command: {cmd_name} with args: {args}")

        if cmd_name == 'ping':
            if not args:
                return None  # No arguments provided
            target = args[0]
            if self.is_valid_domain_or_ip(target):
                return args
            else:
                return None
        else:
            # For other commands, implement as needed
            return None

    def is_valid_domain_or_ip(self, target):
        """
        Validates if the target is a valid domain or IP address.
        """
        logger.info(f"Validating target: {target}")

        import re
        # Simple regex for IP address
        ip_regex = r'^(\d{1,3}\.){3}\d{1,3}$'
        # Simple regex for domain name
        domain_regex = r'^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'

        if re.match(ip_regex, target) or re.match(domain_regex, target):
            return True
        else:
            return False

    def evaluate_response(self, output, criteria):
        """
        Evaluate the command output based on criteria.
        For simplicity, check if criteria is a substring of the output.
        """
        if not criteria:
            return True  # No criteria means automatic success
        return criteria in output

    def self_evaluate(self, task):
        # Allow all tasks of type 'network_ping' for this agent
        logger.info(f"Self evaluating task: {task}")
        return task["type"] == 'network_ping'

    def get_agent_capabilities(self):
        # Return a list of capabilities for this agent
        return ['ping']

    def start(self):
        threading.Thread(target=self.receive).start()

    # Overriding call_openai_api to extract the assistant's reply
    def call_openai_api(self, prompt):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}'
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()
        # Extract the assistant's reply
        assistant_reply = response_data['choices'][0]['message']['content']
        return assistant_reply
