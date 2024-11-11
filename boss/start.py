import logging
import os
import signal
import sys
import threading
from typing import Any, List, Tuple

from boss.boss import BOSS
from boss.events import shutdown_event
from boss.wrappers.wrapper_api_explorer import WrapperAPIExplorer
from boss.wrappers.wrapper_conversation import WrapperConversation
from boss.wrappers.wrapper_dig import DigWrapperAgent
from boss.wrappers.wrapper_get_ssl import WrapperGetSSLCertificateAgent
from boss.wrappers.wrapper_ping_agent import WrapperPing
from boss.wrappers.wrapper_rest import WrapperRESTTestAgent
from boss.wrappers.wrapper_scan_ports import WrapperScanPortAgent
from boss.wrappers.wrapper_websocket import WrapperWebSocketTestAgent
from boss.wrappers.wrapper_whois import WhoisWrapperAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to hold component instances
components: List[Any] = []
threads: List[Tuple[Any, threading.Thread]] = []
is_shutting_down = threading.Event()


def force_exit():
    """
    Force exits the application after a timeout.
    This ensures the app terminates even if some threads are stuck.
    """
    logger.warning("Forcing application exit...")
    os._exit(1)


def signal_handler(signum, frame):
    """
    Enhanced signal handler that ensures proper shutdown on both Unix and Windows.
    """
    if is_shutting_down.is_set():
        logger.warning("Received second interrupt, forcing immediate exit...")
        force_exit()
        return

    logger.info(f"Signal {signum} received. Initiating graceful shutdown.")
    is_shutting_down.set()
    shutdown_event.set()

    # Start a timer to force exit if graceful shutdown takes too long
    force_exit_timer = threading.Timer(5.0, force_exit)
    force_exit_timer.daemon = True
    force_exit_timer.start()


def start_component(component_cls):
    """
    Initializes and starts a component in a separate thread with enhanced error handling.
    """
    try:
        logger.info(f"Initializing component: {component_cls.__name__}")
        component = component_cls()
        components.append(component)
        thread = threading.Thread(
            target=component.start, daemon=True, name=f"{component_cls.__name__}_thread"
        )
        thread.start()
        threads.append((component, thread))
        return True
    except Exception as e:
        logger.error(f"Failed to initialize {component_cls.__name__}: {str(e)}")
        return False


def shutdown():
    """
    Enhanced graceful shutdown with timeout handling.
    """
    if is_shutting_down.is_set():
        return

    is_shutting_down.set()
    shutdown_event.set()
    logger.info("Initiating graceful shutdown of all components...")

    for component, thread in threads:
        try:
            logger.info(f"Stopping component: {thread.name}")
            component.stop()
            thread.join(timeout=2.0)  # Reduced timeout per component

            if thread.is_alive():
                logger.warning(f"Component {thread.name} did not stop gracefully")
        except Exception as e:
            logger.error(f"Error shutting down {thread.name}: {str(e)}")

    logger.info("Shutdown complete")


def main():
    """
    Enhanced main entry point with proper Windows signal handling.
    """
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # On Windows, ensure CTRL+C is properly handled
    if sys.platform == "win32":
        try:
            import win32api

            win32api.SetConsoleCtrlHandler(
                lambda x: signal_handler(signal.SIGINT, None), True
            )
        except ImportError:
            logger.warning(
                "win32api not available, Windows CTRL+C handling may be limited"
            )

    components_to_start = [
        BOSS,
        WrapperPing,
        WrapperConversation,
        WrapperScanPortAgent,
        WrapperGetSSLCertificateAgent,
        WhoisWrapperAgent,
        DigWrapperAgent,
        WrapperRESTTestAgent,
        WrapperWebSocketTestAgent,
        WrapperAPIExplorer,
    ]

    try:
        # Start all components
        for component_cls in components_to_start:
            if not start_component(component_cls):
                logger.error(
                    f"Failed to start {component_cls.__name__}, initiating shutdown"
                )
                shutdown()
                return

        logger.info("All components started successfully")

        # Keep the main thread alive until shutdown is signaled
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1.0)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
