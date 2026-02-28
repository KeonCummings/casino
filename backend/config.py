"""LLM provider configuration — supports Anthropic, OpenAI, and Ollama."""

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    provider: str = ""
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0

    def __post_init__(self):
        self.provider = self.provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.api_key = self.api_key or os.getenv("LLM_API_KEY", "")
        self.model = self.model or os.getenv("LLM_MODEL", "")
        self.base_url = self.base_url or os.getenv("LLM_BASE_URL", "")

        if not self.model:
            self.model = {
                "anthropic": "claude-sonnet-4-20250514",
                "openai": "gpt-4o",
                "ollama": "llama3.1",
            }.get(self.provider, "claude-sonnet-4-20250514")

        if not self.base_url and self.provider == "ollama":
            self.base_url = "http://localhost:11434"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: str = ""
    workspace_root: str = ""
    execution_timeout: int = 30
    host: str = "0.0.0.0"
    port: int = 8000

    def __post_init__(self):
        self.sandbox = self.sandbox or os.getenv("CASINO_SANDBOX", "subprocess")
        self.workspace_root = self.workspace_root or os.getenv(
            "CASINO_WORKSPACE_ROOT", "./workspace"
        )
        self.execution_timeout = int(
            os.getenv("CASINO_EXEC_TIMEOUT", str(self.execution_timeout))
        )


def get_config() -> AppConfig:
    return AppConfig()
