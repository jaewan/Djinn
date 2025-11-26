"""
Message handler registry for TCP server.

Replaces large if/elif chains with a clean registry pattern.
"""

import logging
from typing import Dict, Callable, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class MessageHandlerRegistry:
    """
    Registry for message type handlers.
    
    Replaces large if/elif chains with a clean registration pattern.
    """
    
    def __init__(self):
        """Initialize handler registry."""
        self.handlers: Dict[int, Callable] = {}
        self.handler_names: Dict[int, str] = {}  # For logging
    
    def register(self, message_type: int, handler: Callable, name: Optional[str] = None) -> None:
        """
        Register a handler for a message type.
        
        Args:
            message_type: Message type (integer, e.g., 0x01, 0x02)
            handler: Async handler function
            name: Optional name for logging (defaults to handler.__name__)
        """
        if message_type in self.handlers:
            logger.warning(f"Overwriting handler for message type {message_type:02x}")
        
        self.handlers[message_type] = handler
        self.handler_names[message_type] = name or handler.__name__
        logger.debug(f"Registered handler for message type {message_type:02x} ({self.handler_names[message_type]})")
    
    async def handle(self, message_type: int, *args, **kwargs) -> Any:
        """
        Handle a message by dispatching to the registered handler.
        
        Args:
            message_type: Message type (integer)
            *args: Positional arguments for handler
            **kwargs: Keyword arguments for handler
            
        Returns:
            Handler result
            
        Raises:
            ValueError: If message type is not registered
        """
        handler = self.handlers.get(message_type)
        if handler is None:
            raise ValueError(f"Unknown message type: {message_type:02x}. Registered types: {list(self.handlers.keys())}")
        
        handler_name = self.handler_names.get(message_type, 'unknown')
        logger.debug(f"Dispatching message type {message_type:02x} to {handler_name}")
        
        try:
            return await handler(*args, **kwargs)
        except Exception as e:
            logger.error(f"Handler {handler_name} failed: {e}", exc_info=True)
            raise
    
    def is_registered(self, message_type: int) -> bool:
        """Check if a message type is registered."""
        return message_type in self.handlers
    
    def get_registered_types(self) -> list:
        """Get list of registered message types."""
        return list(self.handlers.keys())
    
    def unregister(self, message_type: int) -> None:
        """Unregister a handler (for testing)."""
        if message_type in self.handlers:
            del self.handlers[message_type]
            del self.handler_names[message_type]
            logger.debug(f"Unregistered handler for message type {message_type:02x}")


# Message type constants
class MessageType:
    """Message type constants for TCP server."""
    EXECUTE_SUBGRAPH = 0x01  # Legacy
    EXECUTE_OPERATION = 0x02  # Legacy
    EXECUTE_SUBGRAPH_COORDINATOR = 0x03  # Coordinator subgraph
    CACHE_QUERY = 0x04  # Cache query
    REGISTER_MODEL = 0x05  # Model registration
    EXECUTE_MODEL = 0x06  # Model execution
    ERROR = 0xFF  # Error response


def create_default_registry() -> MessageHandlerRegistry:
    """
    Create and configure default handler registry.
    
    Returns:
        Configured MessageHandlerRegistry
    """
    registry = MessageHandlerRegistry()
    
    # Register handlers (will be populated by tcp_server.py)
    # This function exists for future extensibility
    
    return registry

