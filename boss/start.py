import logging
import threading
import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from wrappers.wrapper_conversation import WrapperConversation
from wrappers.wrapper_get_ssl import WrapperGetSSLCertificateAgent
from wrappers.wrapper_ping_agent import WrapperPing
from wrappers.wrapper_scan_ports import WrapperScanPortAgent

from boss import BOSS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to hold component instances
components = []
threads = []
shutdown_event = threading.Event()


def start_component(component_cls):
    """
    Initializes and starts a component in a separate thread.
    """
    try:
        logger.info(f"Initializing component: {component_cls.__name__}")
        component = component_cls()
        components.append(component)
        thread = threading.Thread(target=component.start)
        thread.start()
        threads.append((component, thread))
    except Exception as e:
        logger.error(f"Failed to initialize {component_cls.__name__}: {str(e)}")


def shutdown():
    """
    Gracefully shuts down all running components.
    """
    logger.info("Shutting down components...")
    for component, thread in threads:
        component.stop()
    for component, thread in threads:
        thread.join(timeout=5)
    logger.info("All components have been stopped.")


def connect_to_kafka(producer):
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
                time.sleep(retry_delay)
            else:
                logger.error("Failed to connect to Kafka after multiple attempts.")
                raise


def main():
    """
    Main entry point.
    """
    producer = KafkaProducer(bootstrap_servers="127.0.0.1:9092")

    try:
        connect_to_kafka(producer)

        # Start components one at a time to better handle errors
        start_component(BOSS)
        logger.info("BOSS component started successfully")

        start_component(WrapperPing)
        logger.info("Ping agent started successfully")

        start_component(WrapperConversation)
        logger.info("Conversation agent started successfully")

        start_component(WrapperScanPortAgent)
        logger.info("Scan ports agent started successfully")

        start_component(WrapperGetSSLCertificateAgent)
        logger.info("Get SSL agent started successfully")

        # Keep the application running until interrupted
        while not shutdown_event.is_set():
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
