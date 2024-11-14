import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import websockets
from pydantic import BaseModel, Field, ValidationError

from boss.utils import serialize_task_to_string
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class WebSocketMessage(BaseModel):
    """Model for WebSocket message"""

    content: Dict[str, Any] = Field(description="Message content to be sent")
    delay: Optional[float] = Field(
        description="Delay before sending this message (in seconds)"
    )


class WebSocketTestCommand(BaseModel):
    """Model for WebSocket test command parameters"""

    url: str = Field(description="WebSocket endpoint URL")
    messages: List[WebSocketMessage] = Field(
        description="List of messages to send in sequence"
    )
    headers: Dict[str, str] = Field(description="WebSocket connection headers")
    auth_type: Optional[str] = Field(
        description="Authentication type (None, Bearer, JWT, etc.)"
    )
    auth_params: Optional[Dict[str, str]] = Field(
        description="Authentication parameters"
    )
    test_scenario: str = Field(
        description="Test scenario (normal, connection_drop, malformed_message, etc.)"
    )
    timeout: Optional[float] = Field(description="Connection timeout in seconds")
    expected_responses: Optional[List[Dict[str, Any]]] = Field(
        description="Expected response patterns to validate against"
    )


class WrapperWebSocketTestAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_websocket_tester",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_logging()

    def setup_logging(self):
        """Setup task-specific logging"""
        self.task_logger = logging.getLogger(f"{self.agent_id}_task")
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.task_logger.addHandler(handler)
        self.task_logger.setLevel(logging.INFO)

    def _call_openai_api(self, prompt: str) -> WebSocketTestCommand:
        """Call OpenAI API with structured output parsing for WebSocket test generation"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            system_prompt = """
            Extract WebSocket test parameters from the task description. Consider:
            1. Identify the WebSocket endpoint
            2. Determine message sequence and timing
            3. Define authentication requirements
            4. Generate appropriate test scenario
            
            Common test scenarios:
            - normal: Standard WebSocket connection and message exchange
            - connection_drop: Test connection drop handling
            - malformed_message: Send malformed message format
            - reconnect: Test reconnection capabilities
            - concurrent: Test concurrent connections
            - auth_failure: Test authentication failure scenarios
            - large_message: Test with large payload
            - rate_limit: Test rate limiting behavior
            """

            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=WebSocketTestCommand,
            )

            parsed_command = completion.choices[0].message.parsed
            logger.info(f"Parsed command parameters: {parsed_command}")
            return parsed_command

        except ValidationError as ve:
            logger.error(f"Validation error when parsing command: {ve}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            raise

    async def execute_websocket_test(
        self, command: WebSocketTestCommand
    ) -> Dict[str, Any]:
        """Execute WebSocket test with the given parameters"""
        results = []
        start_time = datetime.now(timezone.utc)

        try:
            self.task_logger.info(f"Initiating WebSocket connection to: {command.url}")

            # Prepare headers with authentication if required
            headers = dict(command.headers)
            if command.auth_type and command.auth_params:
                if command.auth_type in ["Bearer", "JWT"]:
                    headers["Authorization"] = (
                        f"Bearer {command.auth_params.get('token', '')}"
                    )

            # Connect to WebSocket server
            async with websockets.connect(
                command.url, extra_headers=headers, timeout=command.timeout
            ) as websocket:
                self.task_logger.info("WebSocket connection established")

                # Handler for receiving messages
                async def receive_messages():
                    try:
                        while True:
                            message = await websocket.recv()
                            timestamp = datetime.now(timezone.utc).isoformat()
                            results.append(
                                {
                                    "type": "received",
                                    "timestamp": timestamp,
                                    "content": message,
                                }
                            )
                            self.task_logger.info(f"Received: {message}")
                    except websockets.exceptions.ConnectionClosed:
                        self.task_logger.info("Connection closed")
                    except Exception as e:
                        self.task_logger.error(f"Error receiving message: {str(e)}")

                # Start receive handler
                receive_task = asyncio.create_task(receive_messages())

                # Process test scenario
                if command.test_scenario == "connection_drop":
                    await asyncio.sleep(2)
                    await websocket.close()
                else:
                    # Send messages according to the test plan
                    for msg in command.messages:
                        if msg.delay:
                            await asyncio.sleep(msg.delay)

                        if command.test_scenario == "malformed_message":
                            # Simulate malformed message
                            await websocket.send("}{invalid json}")
                        else:
                            message_content = json.dumps(msg.content)
                            await websocket.send(message_content)

                            timestamp = datetime.now(timezone.utc).isoformat()
                            results.append(
                                {
                                    "type": "sent",
                                    "timestamp": timestamp,
                                    "content": msg.content,
                                }
                            )
                            self.task_logger.info(f"Sent: {message_content}")

                        # Wait for potential response
                        await asyncio.sleep(0.5)

                # Clean up
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Validate responses if expected patterns are provided
            validation_results = None
            if command.expected_responses:
                validation_results = self._validate_responses(
                    results, command.expected_responses
                )

            return {
                "success": True,
                "execution_time": execution_time,
                "messages": results,
                "scenario": command.test_scenario,
                "validation_results": validation_results,
            }

        except websockets.exceptions.WebSocketException as e:
            self.task_logger.error(f"WebSocket error: {str(e)}")
            return {
                "success": False,
                "error": f"WebSocket error: {str(e)}",
                "scenario": command.test_scenario,
            }
        except Exception as e:
            self.task_logger.error(f"Error during WebSocket test: {str(e)}")
            return {
                "success": False,
                "error": f"Error during WebSocket test: {str(e)}",
                "scenario": command.test_scenario,
            }

    def _validate_responses(
        self, results: List[Dict[str, Any]], expected_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate received messages against expected patterns"""
        validation_results = []
        received_messages = [r for r in results if r["type"] == "received"]

        for pattern in expected_patterns:
            matched = False
            for message in received_messages:
                if self._match_pattern(message["content"], pattern):
                    matched = True
                    break
            validation_results.append({"pattern": pattern, "matched": matched})

        return {
            "total_patterns": len(expected_patterns),
            "matched_patterns": sum(1 for r in validation_results if r["matched"]),
            "details": validation_results,
        }

    def _match_pattern(self, message: Any, pattern: Dict[str, Any]) -> bool:
        """Match a message against an expected pattern"""
        try:
            if isinstance(message, str):
                message = json.loads(message)

            for key, value in pattern.items():
                if key not in message or message[key] != value:
                    return False
            return True
        except (json.JSONDecodeError, TypeError, KeyError):
            return False

    async def process_task_async(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task asynchronously"""
        if not isinstance(task, dict) or "_id" not in task:
            logger.error("Invalid task format")
            return {"success": False, "note": "Invalid task format"}

        try:
            task_id = task.get("_id")
            self.task_logger.info(f"Processing task with ID: {task_id}")

            current_step_index = task.get("current_step_index")
            steps = task.get("steps", [])
            if current_step_index is None or current_step_index >= len(steps):
                logger.error("Invalid current_step_index")
                return {"task_id": task_id, "error": "Invalid current_step_index"}

            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")

            # Generate test parameters using LLM
            parsed_command = self._call_openai_api(step_description)

            # Execute the WebSocket test
            test_result = await self.execute_websocket_test(parsed_command)

            # Prepare the result
            result = {
                "task_id": str(task_id),
                "agent_id": self.agent_id,
                "result": serialize_task_to_string(test_result),
                "metadata": {
                    "url": parsed_command.url,
                    "test_scenario": parsed_command.test_scenario,
                    "auth_type": parsed_command.auth_type,
                },
            }

            return result

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return {
                "task_id": str(task.get("_id", "unknown")),
                "error": f"Validation error: {ve}",
                "note": "Invalid command parameters",
            }
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {
                "task_id": str(task.get("_id", "unknown")),
                "error": str(e),
                "note": "Exception occurred during task execution",
            }

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for process_task_async"""
        return asyncio.run(self.process_task_async(task))
