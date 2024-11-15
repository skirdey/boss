import logging
import os
import signal
import sys
import threading
import time
from typing import Any, List, Tuple

from boss.boss import BOSS
from boss.wrappers.wrapper_whois import WhoisWrapperAgent
from boss.wrappers.wrapper_html import WrapperHTML
from boss.events import shutdown_event
from boss.wrappers.wrapper_api_explorer import WrapperAPIExplorer
from boss.wrappers.wrapper_conversation import WrapperConversation
from boss.wrappers.wrapper_dig import DigWrapperAgent
from boss.wrappers.wrapper_get_ssl import WrapperGetSSLCertificateAgent
from boss.wrappers.wrapper_ping_agent import WrapperPing
from boss.wrappers.wrapper_rest import WrapperRESTTestAgent
from boss.wrappers.wrapper_scan_ports import WrapperScanPortAgent
from boss.wrappers.wrapper_sql_injection_agent import WrapperSQLInjectionAgent
from boss.wrappers.wrapper_websocket import WrapperWebSocketTestAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure logging to print in bright blue
class CustomFormatter(logging.Formatter):
    """Custom logging formatter to print logs in bright blue."""

    BLUE = "\033[94m"
    RESET = "\033[0m"
    FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    def format(self, record):
        log_fmt = self.BLUE + self.FORMAT + self.RESET
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Apply the custom formatter to the root logger
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.getLogger().handlers = [handler]


# Global variables
components: List[Any] = []
threads: List[Tuple[Any, threading.Thread]] = []
is_shutting_down = threading.Event()
main_thread_shutdown = threading.Event()  # New event for main thread control

def force_exit():
    """
    Force exits the application after a timeout.
    """
    logger.warning("Forcing application exit...")
    # Use a more forceful exit method
    os.kill(os.getpid(), signal.SIGKILL)

def signal_handler(signum, frame):
    """
    Enhanced signal handler with better Unix system support.
    """
    # Prevent re-entering the signal handler
    if is_shutting_down.is_set():
        logger.warning("Received second interrupt, forcing immediate exit...")
        force_exit()
        return

    logger.info(f"Signal {signum} received. Initiating graceful shutdown.")
    
    # Set both shutdown flags
    is_shutting_down.set()
    shutdown_event.set()
    main_thread_shutdown.set()  # Signal the main thread to exit

    # Start force exit timer in a separate thread to avoid signal handler complications
    def delayed_force_exit():
        time.sleep(5.0)
        if not all(not thread.is_alive() for _, thread in threads):
            force_exit()
    
    force_exit_thread = threading.Thread(target=delayed_force_exit, daemon=True)
    force_exit_thread.start()

def shutdown():
    """
    Enhanced graceful shutdown with better thread management.
    """
    if is_shutting_down.is_set():
        return

    is_shutting_down.set()
    shutdown_event.set()
    logger.info("Initiating graceful shutdown of all components...")

    # First attempt: graceful shutdown of all components
    for component, thread in threads:
        try:
            logger.info(f"Stopping component: {thread.name}")
            component.stop()
        except Exception as e:
            logger.error(f"Error stopping {thread.name}: {str(e)}")

    # Second phase: wait for threads to finish with timeout
    shutdown_start = time.time()
    remaining_threads = list(threads)
    
    while remaining_threads and (time.time() - shutdown_start) < 4.0:
        for component, thread in remaining_threads[:]:  # Create a copy for safe iteration
            if not thread.is_alive():
                remaining_threads.remove((component, thread))
                logger.info(f"Component {thread.name} stopped successfully")
        time.sleep(0.1)

    # Log warning for any remaining threads
    for component, thread in remaining_threads:
        if thread.is_alive():
            logger.warning(f"Component {thread.name} did not stop gracefully")

    logger.info("Shutdown complete")

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


def main():
    """
    Enhanced main entry point with proper Windows signal handling.
    """
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set up signal handlers with specific handlers for different platforms
    if sys.platform == "win32":
        try:
            import win32api
            win32api.SetConsoleCtrlHandler(lambda x: signal_handler(signal.SIGINT, None), True)
        except ImportError:
            logger.warning("win32api not available, Windows CTRL+C handling may be limited")
            signal.signal(signal.SIGINT, signal_handler)
    else:
        # Unix-specific signal handling setup
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # Add SIGBREAK if available (some Unix systems)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    components_to_start = [
        BOSS,
        WrapperPing,
        WrapperConversation,
        WrapperScanPortAgent,
        # WrapperGetSSLCertificateAgent,
        # DigWrapperAgent,
        WrapperRESTTestAgent,
        # WrapperWebSocketTestAgent,
        WrapperAPIExplorer,
        WrapperSQLInjectionAgent,
        WrapperHTML,
        # WhoisWrapperAgent,

    ]

    try:
        # Start all components
        for component_cls in components_to_start:
            if not start_component(component_cls):
                logger.error(f"Failed to start {component_cls.__name__}, initiating shutdown")
                shutdown()
                return

        logger.info("All components started successfully")

        # Enhanced main loop with better signal handling
        while not main_thread_shutdown.is_set():
            try:
                # Shorter sleep intervals for more responsive shutdown
                main_thread_shutdown.wait(timeout=0.1)
            except KeyboardInterrupt:
                # Handle KeyboardInterrupt explicitly in the main thread
                signal_handler(signal.SIGINT, None)
                break

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
