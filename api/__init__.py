"""
api/__init__.py — API module

Mount routers from here.
"""

from api.kg import router as kg_router

__all__ = ["kg_router"]
