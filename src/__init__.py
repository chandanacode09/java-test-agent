"""Java Test Agent - Automated test generation and fixing for Java/Maven projects."""

from .java_agent import JavaTestAgent, JavaAgentConfig, JavaAgentResult
from .models import Language, TestType, TestCategory, ErrorCategory

__version__ = "1.0.0"
__all__ = [
    "JavaTestAgent",
    "JavaAgentConfig",
    "JavaAgentResult",
    "Language",
    "TestType",
    "TestCategory",
    "ErrorCategory",
]
