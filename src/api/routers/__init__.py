from .compare import router as compare_router
from .inspect import router as inspect_router
from .llm import router as llm_router
from .ml import router as ml_router
from .rag import router as rag_router
from .system import router as system_router

__all__ = [
    "system_router",
    "rag_router",
    "ml_router",
    "llm_router",
    "compare_router",
    "inspect_router",
]
