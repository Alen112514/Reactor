"""
LLM Provider Management Service
Handles multiple LLM providers (ChatGPT, DeepSeek, Grok) with LangChain integration
"""

from enum import Enum
from typing import Dict, Any, Optional, List
import asyncio
import httpx
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
import logging

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Available LLM providers and API services"""
    # OpenAI Models
    OPENAI_GPT4 = "openai-gpt4"
    OPENAI_GPT4O = "openai-gpt4o"
    OPENAI_GPT41 = "openai-gpt4.1"
    OPENAI_GPT_O3 = "openai-gpt-o3"
    OPENAI_GPT35 = "openai-gpt3.5"
    
    # DeepSeek Models
    DEEPSEEK_V2 = "deepseek-v2"
    DEEPSEEK_V3 = "deepseek-v3"
    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_CODER = "deepseek-coder"
    
    # X.AI Models
    GROK_BETA = "grok-beta"
    
    # Web Search Services
    TAVILY = "tavily"

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    model_config = {"protected_namespaces": ()}
    
    provider: LLMProvider
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    cost_per_1k_tokens: float = Field(default=0.002)

class DeepSeekLLM(BaseChatModel):
    """Custom LangChain wrapper for DeepSeek API"""
    
    model_name: str = "deepseek-chat"
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from DeepSeek API"""
        return asyncio.run(self._agenerate(messages, stop, **kwargs))
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate response from DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            else:
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
            **kwargs
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration
        
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

class GrokLLM(BaseChatModel):
    """Custom LangChain wrapper for Grok API (via X.AI)"""
    
    model_name: str = "grok-beta"
    api_key: str
    base_url: str = "https://api.x.ai/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @property
    def _llm_type(self) -> str:
        return "grok"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from Grok API"""
        return asyncio.run(self._agenerate(messages, stop, **kwargs))
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate response from Grok API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            else:
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
            **kwargs
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration
        
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

class LLMProviderService:
    """Service for managing multiple LLM providers"""
    
    def __init__(self, db=None):
        self.db = db
        self.providers: Dict[LLMProvider, LLMConfig] = {}
        self.clients: Dict[LLMProvider, BaseChatModel] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for all providers"""
        self.providers = {
            # OpenAI Models
            LLMProvider.OPENAI_GPT4: LLMConfig(
                provider=LLMProvider.OPENAI_GPT4,
                model_name="gpt-4-turbo-preview",
                cost_per_1k_tokens=0.03,
                max_tokens=4096
            ),
            LLMProvider.OPENAI_GPT4O: LLMConfig(
                provider=LLMProvider.OPENAI_GPT4O,
                model_name="gpt-4o",
                cost_per_1k_tokens=0.005,  # GPT-4o is more cost-effective
                max_tokens=4096
            ),
            LLMProvider.OPENAI_GPT41: LLMConfig(
                provider=LLMProvider.OPENAI_GPT41,
                model_name="gpt-4.1-preview",  # Assuming future model name
                cost_per_1k_tokens=0.035,  # Likely premium pricing
                max_tokens=8192
            ),
            LLMProvider.OPENAI_GPT_O3: LLMConfig(
                provider=LLMProvider.OPENAI_GPT_O3,
                model_name="gpt-o3",  # New reasoning model
                cost_per_1k_tokens=0.06,  # Premium reasoning model
                max_tokens=4096
            ),
            LLMProvider.OPENAI_GPT35: LLMConfig(
                provider=LLMProvider.OPENAI_GPT35,
                model_name="gpt-3.5-turbo",
                cost_per_1k_tokens=0.002,
                max_tokens=4096
            ),
            
            # DeepSeek Models
            LLMProvider.DEEPSEEK_V2: LLMConfig(
                provider=LLMProvider.DEEPSEEK_V2,
                model_name="deepseek-chat",
                cost_per_1k_tokens=0.0014,
                max_tokens=4096,
                base_url="https://api.deepseek.com/v1"
            ),
            LLMProvider.DEEPSEEK_V3: LLMConfig(
                provider=LLMProvider.DEEPSEEK_V3,
                model_name="deepseek-v3",
                cost_per_1k_tokens=0.0027,  # V3 likely has higher cost
                max_tokens=8192,  # Larger context window
                base_url="https://api.deepseek.com/v1"
            ),
            LLMProvider.DEEPSEEK_R1: LLMConfig(
                provider=LLMProvider.DEEPSEEK_R1,
                model_name="deepseek-r1",
                cost_per_1k_tokens=0.008,  # R1 reasoning model premium
                max_tokens=4096,
                base_url="https://api.deepseek.com/v1"
            ),
            LLMProvider.DEEPSEEK_CODER: LLMConfig(
                provider=LLMProvider.DEEPSEEK_CODER,
                model_name="deepseek-coder",
                cost_per_1k_tokens=0.0014,
                max_tokens=4096,
                base_url="https://api.deepseek.com/v1"
            ),
            
            # X.AI Models
            LLMProvider.GROK_BETA: LLMConfig(
                provider=LLMProvider.GROK_BETA,
                model_name="grok-beta",
                cost_per_1k_tokens=0.005,
                max_tokens=4096,
                base_url="https://api.x.ai/v1"
            )
        }
    
    def configure_provider(self, provider: LLMProvider, config: LLMConfig):
        """Configure a specific LLM provider"""
        self.providers[provider] = config
        logger.info(f"Configured provider {provider.value}")
    
    def get_llm_client(self, provider: LLMProvider, api_key: str) -> BaseChatModel:
        """Get LangChain LLM client for specified provider"""
        if provider in self.clients:
            return self.clients[provider]
        
        config = self.providers.get(provider)
        if not config:
            raise ValueError(f"Provider {provider.value} not configured")
        
        # Create appropriate LangChain client
        if provider in [
            LLMProvider.OPENAI_GPT4, 
            LLMProvider.OPENAI_GPT4O, 
            LLMProvider.OPENAI_GPT41, 
            LLMProvider.OPENAI_GPT_O3, 
            LLMProvider.OPENAI_GPT35
        ]:
            client = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=api_key
            )
        elif provider in [
            LLMProvider.DEEPSEEK_V2, 
            LLMProvider.DEEPSEEK_V3, 
            LLMProvider.DEEPSEEK_R1, 
            LLMProvider.DEEPSEEK_CODER
        ]:
            client = DeepSeekLLM(
                model_name=config.model_name,
                api_key=api_key,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        elif provider == LLMProvider.GROK_BETA:
            client = GrokLLM(
                model_name=config.model_name,
                api_key=api_key,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {provider.value}")
        
        self.clients[provider] = client
        return client
    
    async def get_llm_client_for_user(
        self, 
        provider: LLMProvider, 
        session_id: str, 
        db
    ) -> BaseChatModel:
        """
        Get LangChain LLM client using user's stored API key
        """
        from app.services.api_key_manager import api_key_manager
        
        # Get user's API key for this provider
        api_key = await api_key_manager.get_api_key(db, session_id, provider.value)
        
        if not api_key:
            raise ValueError(f"No valid API key found for provider {provider.value} and session {session_id}")
        
        # Create a new client instance with user's API key (don't cache user-specific clients)
        config = self.providers.get(provider)
        if not config:
            raise ValueError(f"Provider {provider.value} not configured")
        
        # Create appropriate LangChain client
        if provider in [
            LLMProvider.OPENAI_GPT4, 
            LLMProvider.OPENAI_GPT4O, 
            LLMProvider.OPENAI_GPT41, 
            LLMProvider.OPENAI_GPT_O3, 
            LLMProvider.OPENAI_GPT35
        ]:
            client = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=api_key
            )
        elif provider in [
            LLMProvider.DEEPSEEK_V2, 
            LLMProvider.DEEPSEEK_V3, 
            LLMProvider.DEEPSEEK_R1, 
            LLMProvider.DEEPSEEK_CODER
        ]:
            client = DeepSeekLLM(
                model_name=config.model_name,
                api_key=api_key,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        elif provider == LLMProvider.GROK_BETA:
            client = GrokLLM(
                model_name=config.model_name,
                api_key=api_key,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {provider.value}")
        
        return client
    
    def get_provider_info(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get information about a specific provider"""
        config = self.providers.get(provider)
        if not config:
            return {}
        
        return {
            "provider": provider.value,
            "model_name": config.model_name,
            "cost_per_1k_tokens": config.cost_per_1k_tokens,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all available providers with their info"""
        return [
            self.get_provider_info(provider) 
            for provider in LLMProvider
        ]
    
    def estimate_cost(self, provider: LLMProvider, token_count: int) -> float:
        """Estimate cost for token usage with specific provider"""
        config = self.providers.get(provider)
        if not config:
            return 0.0
        
        return (token_count / 1000) * config.cost_per_1k_tokens
    
    def select_optimal_provider(
        self, 
        task_type: str, 
        budget_limit: float, 
        token_estimate: int
    ) -> LLMProvider:
        """Select optimal provider based on task type and budget"""
        
        # Filter providers by budget
        affordable_providers = []
        for provider in LLMProvider:
            cost = self.estimate_cost(provider, token_estimate)
            if cost <= budget_limit:
                affordable_providers.append((provider, cost))
        
        if not affordable_providers:
            # Return cheapest option if budget is too low
            return min(
                LLMProvider, 
                key=lambda p: self.providers[p].cost_per_1k_tokens
            )
        
        # Select based on task type
        if task_type in ["coding", "technical", "debugging"]:
            # Prefer DeepSeek Coder, then DeepSeek V3 for coding tasks
            for provider, _ in affordable_providers:
                if provider == LLMProvider.DEEPSEEK_CODER:
                    return provider
            for provider, _ in affordable_providers:
                if provider == LLMProvider.DEEPSEEK_V3:
                    return provider
        
        elif task_type in ["reasoning", "complex_analysis", "math"]:
            # Prefer reasoning models: GPT-o3, DeepSeek-R1, GPT-4.1
            for provider, _ in affordable_providers:
                if provider == LLMProvider.OPENAI_GPT_O3:
                    return provider
            for provider, _ in affordable_providers:
                if provider == LLMProvider.DEEPSEEK_R1:
                    return provider
            for provider, _ in affordable_providers:
                if provider == LLMProvider.OPENAI_GPT41:
                    return provider
        
        elif task_type in ["creative", "general_analysis"]:
            # Prefer GPT-4o (cost-effective), then GPT-4
            for provider, _ in affordable_providers:
                if provider == LLMProvider.OPENAI_GPT4O:
                    return provider
            for provider, _ in affordable_providers:
                if provider == LLMProvider.OPENAI_GPT4:
                    return provider
        
        elif task_type in ["general", "qa", "simple"]:
            # Use most cost-effective option
            return min(affordable_providers, key=lambda x: x[1])[0]
        
        # Default: return most cost-effective
        return min(affordable_providers, key=lambda x: x[1])[0]
    
    async def chat_completion(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        db=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Basic chat completion without tools using user-provided API keys
        """
        try:
            import openai
            from app.core.config import settings
            from app.services.api_key_manager import api_key_manager
            
            # Get user's API key if session_id provided
            api_key = None
            if session_id and db:
                api_key = await api_key_manager.get_api_key(db, session_id, provider.value)
            
            # Fallback to system API key if available
            if not api_key and provider.value.startswith("openai"):
                api_key = settings.OPENAI_API_KEY
            
            if not api_key:
                raise ValueError(f"No API key available for provider {provider.value}")
            
            # Handle different providers
            if provider.value.startswith("openai"):
                client = openai.AsyncOpenAI(api_key=api_key)
                
                config = self.providers.get(provider)
                model_name = config.model_name if config else "gpt-4-turbo-preview"
                
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 4096)
                )
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            
            elif provider.value.startswith("deepseek"):
                # DeepSeek API implementation
                import httpx
                
                config = self.providers.get(provider)
                model_name = config.model_name if config else "deepseek-chat"
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 4096)
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "content": data["choices"][0]["message"]["content"],
                            "usage": data.get("usage", {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            })
                        }
                    else:
                        raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            elif provider.value.startswith("grok"):
                # Grok API implementation (using OpenAI-compatible format)
                import httpx
                
                config = self.providers.get(provider)
                model_name = config.model_name if config else "grok-beta"
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 4096)
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "content": data["choices"][0]["message"]["content"],
                            "usage": data.get("usage", {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            })
                        }
                    else:
                        raise Exception(f"Grok API error: {response.status_code} - {response.text}")
            
            else:
                # Unsupported provider
                return {"content": f"Provider {provider.value} is not yet supported. Please use OpenAI, DeepSeek, or Grok providers.", "usage": {}}
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return {"content": f"Error: {str(e)}", "error": str(e)}
    
    async def chat_with_tools(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        db=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion with function/tool calling support using user-provided API keys
        """
        try:
            import openai
            from app.core.config import settings
            from app.services.api_key_manager import api_key_manager
            
            # Get user's API key if session_id provided
            api_key = None
            if session_id and db:
                api_key = await api_key_manager.get_api_key(db, session_id, provider.value)
            
            # Fallback to system API key if available
            if not api_key and provider.value.startswith("openai"):
                api_key = settings.OPENAI_API_KEY
            
            if not api_key:
                raise ValueError(f"No API key available for provider {provider.value}")
            
            # For now, primarily support OpenAI models which have robust function calling
            if provider.value.startswith("openai"):
                client = openai.AsyncOpenAI(api_key=api_key)
                
                config = self.providers.get(provider)
                model_name = config.model_name if config else "gpt-4-turbo-preview"
                
                # Convert our tool format to OpenAI format
                openai_tools = []
                for tool in tools:
                    openai_tools.append({
                        "type": "function",
                        "function": tool["function"]
                    })
                
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 4096)
                )
                
                message = response.choices[0].message
                
                result = {
                    "content": message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
                # Add tool calls if present
                if message.tool_calls:
                    result["tool_calls"] = []
                    for tool_call in message.tool_calls:
                        result["tool_calls"].append({
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                
                return result
                
            else:
                # For other providers, fall back to regular chat
                # TODO: Implement function calling for DeepSeek, Grok when available
                logger.warning(f"Function calling not fully implemented for {provider.value}, falling back to regular chat")
                return await self.chat_completion(provider, messages, session_id=session_id, db=db, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in chat with tools: {e}")
            return {"content": f"Error: {str(e)}", "error": str(e)}

# Global instance
llm_provider_service = LLMProviderService()