import asyncio
import logging
import sys

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from boss import BOSS
from wrappers.wrapper_conversation import WrapperConversation
from wrappers.wrapper_ping_agent import WrapperPing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to hold component instances
components = []


async def start_component(component_cls):
    """
    Initializes and starts a component in a separate thread.
    """
    try:
        logger.info(f"Initializing component: {component_cls.__name__}")
        component = component_cls()
        components.append(component)
        await asyncio.to_thread(component.start)
    except Exception as e:
        logger.error(f"Failed to initialize {component_cls.__name__}: {str(e)}")
        # Do not re-raise the exception; allow the application to continue
        # raise  # Commented out to prevent stopping the application


async def shutdown():
    """
    Gracefully shuts down all running components.
    """
    logger.info("Shutting down components...")
    shutdown_tasks = [asyncio.to_thread(component.stop) for component in components]
    await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    logger.info("All components have been signaled to stop.")


async def connect_to_kafka(producer):
    """
    Attempts to connect to Kafka with retry logic.
    """
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            producer.send("test-topic", b"Test message")
            producer.flush()
            logger.info("Connected to Kafka successfully.")
            return
        except NoBrokersAvailable:
            if attempt < max_retries:
                logger.warning(
                    f"Kafka brokers not available. Retrying in {retry_delay} seconds... (Attempt {attempt}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Failed to connect to Kafka after multiple attempts.")
                raise


async def main():
    """
    Main entry point for the asyncio event loop.
    """
    producer = KafkaProducer(bootstrap_servers="127.0.0.1:9092")

    try:
        await connect_to_kafka(producer)

        # Start components one at a time to better handle errors
        await start_component(BOSS)
        logger.info("BOSS component started successfully")

        await start_component(WrapperPing)
        logger.info("Ping agent started successfully")

        await start_component(WrapperConversation)
        logger.info("Conversation agent started successfully")

        # await start_component(WrapperNetworkScan)
        # logger.info("Network scan agent started successfully")

        # Keep the application running until interrupted
        while True:
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        await shutdown()


if __name__ == "__main__":
    try:
        # Run the main coroutine using asyncio.run()
        asyncio.run(main())
    except KeyboardInterrupt:
        # This ensures that if KeyboardInterrupt is not caught within main(), it is handled here.
        logger.info("\nShutdown requested...exiting")
        sys.exit(0)
