import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from boss.boss_prompts import BossPrompts
from boss.models import (
    TaskState,
)

TASKS = []

# Load environment variables
load_dotenv()

# Define ANSI escape codes for colors
RESET = "\x1b[0m"
RED = "\x1b[31;1m"


class CustomFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages based on their severity.
    """

    FORMATS = {
        logging.DEBUG: logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s"
        ),
        logging.INFO: logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s"
        ),
        logging.WARNING: logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s"
        ),
        logging.ERROR: logging.Formatter(
            f"{RED}%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s{RESET}"
        ),
        logging.CRITICAL: logging.Formatter(
            f"{RED}%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s{RESET}"
        ),
    }

    def format(self, record):
        formatter = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        return formatter.format(record)


# Configure logging with the custom formatter
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

logger = logging.getLogger(__name__)


class BOSS:
    def __init__(
        self,
        db_uri="mongodb://localhost:27017/",
        kafka_bootstrap_servers="localhost:9092",
        check_interval=1,
        task_queue: asyncio.Queue = None,
        result_queue: asyncio.Queue = None,
        selfplay_response_queue: asyncio.Queue = None,
    ):
        self.task_queue = task_queue or asyncio.Queue()
        self.result_queue = result_queue or asyncio.Queue()
        self.selfplay_response_queue = selfplay_response_queue or asyncio.Queue()

        self.kafka_bootstrap_servers = kafka_bootstrap_servers

        self.task_update_callbacks = []
        self.prompts = BossPrompts()

        self.producer = AIOKafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, indent=2, default=str).encode(
                "utf-8"
            ),
        )
        self.check_interval = check_interval
        self.running = True

        try:
            self.mongo_client = AsyncIOMotorClient(
                db_uri, serverSelectionTimeoutMS=5000
            )
            # Test MongoDB connection

            self.db = self.mongo_client["task_db"]
            self.tasks_collection = self.db["tasks"]
            self.agents_collection = self.db["agents"]
            self.task_history_collection = self.db["task_history"]

        except Exception as err:
            logger.error(f"Error: Could not connect to MongoDB server: {err}")
            raise

    async def _get_agent_result_topics(self):
        """
        Fetch agent result topics asynchronously.
        """
        agents = await self.agents_collection.find().to_list(None)
        return [f"results_from_{agent['agent_id']}" for agent in agents]

    def register_task_update_callback(self, callback):
        self.task_update_callbacks.append(callback)

    async def consume_agent_results(self):
        try:
            async for message in self.result_consumer:
                result = message.value
                # Process the agent result
                await self.handle_agent_result(result)
                # Pass the result to SelfPlay
                await self.result_queue.put(result)
        except Exception as e:
            logger.error(f"Error consuming agent results: {e}")
        finally:
            await self.result_consumer.stop()

    async def handle_agent_result(self, result):
        """
        Handle the result received from an agent.
        Update task status and perform any necessary actions based on the result.
        """
        task_id = result.get("task_id")
        step_id = result.get("step_id")
        success = result.get("success", False)
        error = result.get("error")

        if not task_id or not step_id:
            logger.error("Invalid result format: missing task_id or step_id")
            return

        try:
            # Update task status based on success or failure
            if success:
                logger.info(
                    f"Agent reported successful completion of task {task_id}, step {step_id}"
                )
                await self.tasks_collection.update_one(
                    {"_id": ObjectId(task_id), "tree_structure.children.step_id": step_id},
                    {
                        "$set": {
                            "tree_structure.children.$.status": "Completed",
                        }
                    },
                )

            else:
                logger.warning(
                    f"Agent reported failure for task {task_id}, step {step_id}: {error}"
                )
                await self.tasks_collection.update_one(
                    {"_id": ObjectId(task_id), "tree_structure.children.step_id": step_id},
                    {
                        "$set": {
                            "tree_structure.children.$.status": "Failed",
                        }
                    },
                )

        except Exception as e:
            logger.error(f"Error updating task status: {e}")

    async def _process_task_with_inference(self, task):
        # Placeholder for actual implementation
        logger.info(f"Processing task with inference: {task}")
        # Implement your logic here
        pass

    async def handle_selfplay_responses(self):
        while self.running:
            response = await self.selfplay_response_queue.get()
            try:
                task_id = response["task_id"]
                step_id = response["step_id"]
                agent_id = response["agent_id"]
                targets = response.get("targets", [])
                step_description = response.get("step_description", "")

                logger.info(f"Handling selfplay response: {response}")

                # Update the task in the database with the new step and agent
                await self._update_task_with_step_and_agent(task_id, step_id, agent_id)

                # Send the step to the agent
                await self._send_current_step_to_agent(
                    task_id, step_id, step_description, agent_id, targets
                )
            except Exception as e:
                logger.error(f"Error handling selfplay response: {e}")

    async def _update_task_with_step_and_agent(self, task_id, step_id, agent_id):
        # Ensure tree_structure.children exists
        await self.tasks_collection.update_one(
            {"_id": ObjectId(task_id)},
            {"$setOnInsert": {"tree_structure": {"children": []}}},
            upsert=True,
        )

        # Now perform the array update
        try:
            update_result = await self.tasks_collection.update_one(
                {"_id": ObjectId(task_id), "tree_structure.children.step_id": step_id},
                {
                    "$set": {
                        "status": "In_Progress",
                        "tree_structure.children.$.agent_id": agent_id,
                        "tree_structure.children.$.status": "Assigned",
                    }
                },
            )

            if update_result.modified_count > 0:
                logger.info(
                    f"Successfully updated task {task_id} with step {step_id} and agent {agent_id}."
                )
            else:
                logger.warning(
                    f"No documents were updated for task {task_id} with step {step_id}. Ensure the step_id matches an existing child."
                )
        except Exception as e:
            logger.error(
                f"Failed to update task {task_id} with step {step_id} and agent {agent_id}: {e}"
            )

    async def _send_current_step_to_agent(
        self, task_id, step_id, step_description, agent_id, targets
    ):
        logger.info(
            f"Sending current step {step_id} for task {task_id} to agent {agent_id}"
        )

        # Retrieve agent details (e.g., Kafka topic) from the database
        agent = await self.agents_collection.find_one({"agent_id": agent_id})
        if not agent:
            logger.error(f"Agent {agent_id} not found in the database.")
            return

        kafka_topic = f"tasks_for_{agent_id}"

        if not kafka_topic:
            logger.error(f"Agent {agent_id} does not have a configured Kafka topic.")
            return

        # Prepare the message payload
        payload = {
            "task_id": task_id,
            "step_id": step_id,
            "description": step_description,
            "evaluation_criteria": "Any information or entry points are discovered about the target",
            "targets": targets,
        }

        try:
            # Send the message to the specified Kafka topic
            await self.producer.send(kafka_topic, payload)
            logger.info(
                f"Successfully sent step {step_id} to agent {agent_id} on topic '{kafka_topic}'."
            )

        except Exception as e:
            logger.error(f"Failed to send step {step_id} to agent {agent_id}: {e}")

    async def _listen_step_completion_from_agent(self):
        # listen on kafka consumer for results_from_<agent_id>
        pass

    async def check_tasks(self):
        while self.running:
            logger.info("Checking tasks...")
            try:
                async for task in self.tasks_collection.find(
                    {"status": {"$in": [TaskState.CREATED.value]}}
                ):
                    logger.info(f"Found task: {task.get('_id')}")

                    await self.task_queue.put(task)
                    # Update task status to prevent reprocessing
                    await self.tasks_collection.update_one(
                        {"_id": task["_id"]},
                        {"$set": {"status": TaskState.IN_PROGRESS.value}},
                    )
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error checking tasks: {e}")
                await asyncio.sleep(self.check_interval)

    async def process_tasks(self):
        while self.running:
            task = await self.task_queue.get()
            print(f"Processing task: {task.get('_id')}")
            # Process the task

    async def start(self):
        """Start the BOSS service with asyncio tasks."""
        logger.info("Starting BOSS service...")
        self.running = True

        # Start Kafka producer
        await self.producer.start()
        logger.info("Kafka producer started.")

        # Fetch topics asynchronously
        agent_result_topics = await self._get_agent_result_topics()

        self.result_consumer = AIOKafkaConsumer(
            *agent_result_topics,
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
            group_id="boss_group",
        )
        await self.result_consumer.start()
        logger.info("Kafka consumer started.")

        # Create all tasks
        tasks = [
            self.process_tasks(),
            self.consume_agent_results(),
            self.check_tasks(),
            self.handle_selfplay_responses(),
        ]

        # Start tasks with gather and handle exceptions
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"An unexpected exception occurred: {e}")
        else:
            # Check for exceptions in task results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"A task failed with exception: {result}")

        logger.info("BOSS service started successfully.")

    async def stop(self):
        """Stop the BOSS service gracefully."""
        logger.info("Stopping BOSS service...")
        self.running = False

        # Stop Kafka producer
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer stopped.")

        # Stop Kafka consumer if initialized
        if self.result_consumer:
            await self.result_consumer.stop()
            logger.info("Kafka consumer stopped.")

        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB client closed.")

        if TASKS:
            await asyncio.gather(*TASKS, return_exceptions=True)

        logger.info("BOSS service stopped.")
