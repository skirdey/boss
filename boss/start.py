import asyncio
import logging
import os
import signal
from typing import Any, List

from boss.boss import BOSS
from boss.self_play import SelfPlayMCTS
from boss.wrappers.wrapper_dig_agent import WrapperDigAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
components: List[Any] = []
tasks: List[asyncio.Task] = []
is_shutting_down = asyncio.Event()


def handle_signal(sig):
    logger.info(f"Signal {sig} received. Initiating graceful shutdown.")
    # Schedule the shutdown in the event loop
    asyncio.get_event_loop().call_soon_threadsafe(is_shutting_down.set)


async def start_component(component_cls, loop, *args, **kwargs):
    try:
        logger.info(f"Initializing component: {component_cls.__name__}")
        component = component_cls(*args, **kwargs)
        components.append(component)

        if asyncio.iscoroutinefunction(component.start):
            task = loop.create_task(component.start(), name=component_cls.__name__)
        else:
            task = loop.run_in_executor(None, component.start)
        tasks.append(task)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize {component_cls.__name__}: {str(e)}")
        return False


async def main():
    loop = asyncio.get_running_loop()

    if os.name == "nt":
        # On Windows, rely on KeyboardInterrupt instead of custom signal handling
        pass
    else:
        # Unix: Use loop.add_signal_handler
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
            except NotImplementedError:
                logger.warning(f"Signal {sig} not implemented on this platform.")

    # Initialize queues
    task_queue = asyncio.Queue()
    result_queue = asyncio.Queue()
    selfplay_response_queue = asyncio.Queue()

    components_to_start = [
        (
            BOSS,
            {
                "task_queue": task_queue,
                "result_queue": result_queue,
                "selfplay_response_queue": selfplay_response_queue,
            },
        ),
        (
            SelfPlayMCTS,
            {
                "task_queue": task_queue,
                "result_queue": result_queue,
                "selfplay_response_queue": selfplay_response_queue,
            },
        ),
        # WrapperPing,
        # WrapperConversation,
        WrapperDigAgent,
        # Add other components as needed
    ]

    # Start all components
    for component_info in components_to_start:
        if isinstance(component_info, tuple):
            component_cls, kwargs = component_info
        else:
            component_cls = component_info
            kwargs = {}

        if not await start_component(component_cls, loop, **kwargs):
            logger.error(
                f"Failed to start {component_cls.__name__}, initiating shutdown"
            )
            is_shutting_down.set()
            break

    logger.info("All components started successfully")

    try:
        # Wait for shutdown event
        await is_shutting_down.wait()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Initiating shutdown...")
        is_shutting_down.set()

    # Initiate graceful shutdown
    logger.info("Initiating graceful shutdown...")
    shutdown_tasks = [
        component.stop() for component in components if hasattr(component, "stop")
    ]
    await asyncio.gather(*shutdown_tasks, return_exceptions=True)

    # Cancel all remaining tasks
    for task in tasks:
        task.cancel()

    # Wait for all tasks to finish
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
