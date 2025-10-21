"""Language model configuration and initialization utilities.
"""

from __future__ import annotations
import dspy

from dataclasses import dataclass, field, asdict, is_dataclass
from dataclasses import fields as dc_fields
from enum import IntEnum
from typing import Optional, Any, Dict


# ---------- Provider enumeration ----------
class LanguageModelProvider(IntEnum):
    """Enumeration of supported language model providers.

    This enum defines the available LM providers with numeric values
    for compatibility with protobuf-style serialization.
    """
    LANGUAGE_MODEL_PROVIDER_INVALID = 0      # Invalid/unset provider
    LANGUAGE_MODEL_PROVIDER_AZURE_OPENAI = 1  # Azure OpenAI service
    LANGUAGE_MODEL_PROVIDER_OPENAI = 2        # Direct OpenAI API
    LANGUAGE_MODEL_PROVIDER_LITELLM_SERVER = 3  # LiteLLM proxy server


# ---------- Provider-specific configuration classes ----------
@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI service.

    Attributes:
        api_key: Azure OpenAI API key
        api_base: Base URL for the Azure OpenAI endpoint
        api_version: API version string (e.g., '2023-05-15')
    """
    api_key: str
    api_base: str
    api_version: str


@dataclass
class OpenAIConfig:
    """Configuration for direct OpenAI API access.

    Attributes:
        api_key: OpenAI API key
    """
    api_key: str


@dataclass
class LiteLLMServerConfig:
    """Configuration for LiteLLM proxy server.

    Attributes:
        api_key: API key for the LiteLLM server
        api_base: Base URL for the LiteLLM server endpoint
    """
    api_key: str
    api_base: str


# ---------- Main configuration class with oneof pattern ----------
@dataclass
class LanguageModelProviderConfig:
    """Comprehensive configuration for language model providers.
    """
    provider: LanguageModelProvider
    model_name: str
    temperature: float
    top_p: Optional[float] = None
    max_tokens: int = 0
    parallel_threads: Optional[int] = None
    reasoning_effort: Optional[str] = None

    # oneof config { azure_openai_config | openai_config | litellm_server_config }
    azure_openai_config: Optional[AzureOpenAIConfig] = None
    openai_config: Optional[OpenAIConfig] = None
    litellm_server_config: Optional[LiteLLMServerConfig] = None

    def __post_init__(self) -> None:
        """Validate the oneof constraint after initialization.

        Ensures that at most one provider-specific configuration is set,
        following the protobuf oneof pattern.

        Raises:
            ValueError: If multiple provider configs are set simultaneously
        """
        # Count how many provider configs are non-None
        configs_set = sum(
            x is not None
            for x in (
                self.azure_openai_config,
                self.openai_config,
                self.litellm_server_config,
            )
        )
        if configs_set > 1:
            raise ValueError("Exactly one of the oneof config fields may be set.")
        # It's acceptable to have none set (matches protobuf oneof behavior)

    # ----- Public API: dictionary serialization -----
    def to_dict(
        self,
        *,
        proto_json: bool = True,
        enum_as_value: bool = False,
        omit_none: bool = True,
    ) -> Dict[str, Any]:
        """Serialize configuration to a dictionary with flexible formatting options.

        Args:
            proto_json: If True, use lowerCamelCase keys (protobuf JSON style)
            enum_as_value: If True, use numeric enum values; if False, use names
            omit_none: If True, exclude None/unset fields from output

        Returns:
            Dictionary representation of the configuration
        """
        def snake_to_camel(s: str) -> str:
            """Convert snake_case to lowerCamelCase."""
            parts = s.split('_')
            return parts[0] + ''.join(p.capitalize() for p in parts[1:])

        def transform(obj: Any) -> Any:
            """Recursively transform objects to dictionary format."""
            # Handle dataclass objects
            if is_dataclass(obj):
                raw = {}
                for k, v in asdict(obj).items():
                    # Skip oneof fields here - they're handled separately below
                    if k in {
                        "azure_openai_config",
                        "openai_config",
                        "litellm_server_config",
                    }:
                        continue

                    # Skip None values if requested
                    if omit_none and v is None:
                        continue

                    # Convert key format
                    key = snake_to_camel(k) if proto_json else k

                    # Handle enum values
                    if isinstance(getattr(self, k, None), IntEnum) and isinstance(v, int):
                        raw[key] = int(v) if enum_as_value else LanguageModelProvider(v).name
                        continue

                    raw[key] = transform(v)
                return raw

            # Handle other types
            if isinstance(obj, IntEnum):
                return int(obj) if enum_as_value else obj.name
            if isinstance(obj, list):
                return [transform(x) for x in obj]
            if isinstance(obj, dict):
                return {(snake_to_camel(k) if proto_json else k): transform(v) for k, v in obj.items()}
            return obj

        # Start with the base object transformation
        base = transform(self)

        # Handle oneof provider configs: include only the populated one
        oneof_map = [
            ("azure_openai_config", self.azure_openai_config),
            ("openai_config", self.openai_config),
            ("litellm_server_config", self.litellm_server_config),
        ]

        for name, value in oneof_map:
            key = snake_to_camel(name) if proto_json else name
            if value is not None:
                # Include the populated config
                base[key] = transform(value)
            else:
                # Remove None configs if omit_none is True
                if key in base and omit_none:
                    del base[key]

        return base


def init_lm(lm_config: LanguageModelProviderConfig) -> "dspy.LM":
    """Initialize a DSPy language model instance from configuration.
    """
    # ---- Build base parameters common to all providers ----
    lm_params = {
        "model": lm_config.model_name,
        "temperature": lm_config.temperature,
        "max_tokens": lm_config.max_tokens,
        "timeout": 60,   # Prevent hanging indefinitely on failed requests
        "cache": True,   # Enable response caching for efficiency
    }
    # Add optional parameters if specified
    if lm_config.top_p is not None:
        lm_params["top_p"] = lm_config.top_p

    # ---- Detect which provider config is active (oneof pattern) ----
    configs = {
        "azure_openai_config": lm_config.azure_openai_config,
        "openai_config": lm_config.openai_config,
        "litellm_server_config": lm_config.litellm_server_config,
    }
    set_fields = [name for name, val in configs.items() if val is not None]

    # Validate oneof constraint
    if not set_fields:
        raise ValueError("No provider configuration found in oneof field.")
    if len(set_fields) > 1:
        raise ValueError(f"Multiple provider configs set in oneof: {set_fields}")

    # Extract the active configuration
    config_field = set_fields[0]
    provider_config = configs[config_field]

    # ---- Apply Azure-specific naming convention for LiteLLM ----
    # LiteLLM expects Azure models to be prefixed with 'azure/'
    if lm_config.provider == LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_AZURE_OPENAI:
        if not lm_params["model"].startswith("azure/"):
            lm_params["model"] = f"azure/{lm_params['model']}"

    # ---- Copy provider-specific configuration fields ----
    # Use dataclass field reflection to dynamically copy all non-empty fields
    for f in dc_fields(provider_config):
        value = getattr(provider_config, f.name)

        # Skip None values
        if value is None:
            continue
        # Skip empty strings
        if isinstance(value, str) and value.strip() == "":
            continue

        # Copy the field value to LM parameters
        lm_params[f.name] = value

    # ---- Add optional advanced parameters if specified ----
    if lm_config.parallel_threads is not None:
        lm_params["parallel_threads"] = lm_config.parallel_threads
    if lm_config.reasoning_effort is not None:
        lm_params["reasoning_effort"] = lm_config.reasoning_effort

    # ---- Construct the LM instance and perform validation ----
    lm = dspy.LM(**lm_params)

    # Perform a basic test call to ensure the LM is working
    try:
        _ = lm("hi")  # Simple test prompt
    except Exception as e:
        raise ValueError(f"Failed to initialize language model: {e}") from e

    return lm


def get_lm_cost(lm) -> float:
    """Calculate the total cost of API calls made by a language model.

    This function sums up the cost of all API calls recorded in the LM's
    history. Useful for tracking expenses during development and production.

    Args:
        lm: DSPy language model instance with history tracking

    Returns:
        Total cost in USD (or provider's currency unit)
    """
    return sum([entry.get("cost", 0.0) for entry in lm.history])
