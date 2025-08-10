"""
LLM Provider Management API endpoints
Allows users to select and configure different LLM providers
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.services.llm_provider import llm_provider_service, LLMProvider, LLMConfig
from app.services.api_key_manager import api_key_manager
from app.types import ApiResponse

router = APIRouter()

class LLMProviderInfo(BaseModel):
    """Information about an LLM provider"""
    provider: str
    model_name: str
    cost_per_1k_tokens: float
    max_tokens: int
    temperature: float
    description: str
    strengths: List[str]
    use_cases: List[str]

class LLMSelectionRequest(BaseModel):
    """Request for LLM provider selection"""
    task_type: str = Field(description="Type of task: coding, creative, analysis, general")
    budget_limit: float = Field(description="Budget limit for the task")
    token_estimate: int = Field(description="Estimated token usage")
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

class LLMSelectionResponse(BaseModel):
    """Response with recommended LLM provider"""
    recommended_provider: str
    reasoning: str
    cost_estimate: float
    alternatives: List[Dict[str, Any]]

class LLMConfigRequest(BaseModel):
    """Request to configure an LLM provider"""
    provider: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

class APIKeyRequest(BaseModel):
    """Request to store an API key"""
    session_id: str = Field(description="User session ID")
    provider: str = Field(description="LLM provider name")
    api_key: str = Field(description="User's API key", min_length=10)

class APIKeyValidationRequest(BaseModel):
    """Request to validate an API key"""
    session_id: str = Field(description="User session ID")
    provider: str = Field(description="LLM provider name")

class APIKeyResponse(BaseModel):
    """Response for API key operations"""
    provider: str
    is_valid: Optional[bool]
    last_validated: Optional[str]
    expires_at: str
    needs_validation: bool

@router.get("/providers", response_model=ApiResponse[List[LLMProviderInfo]])
async def list_llm_providers() -> ApiResponse[List[LLMProviderInfo]]:
    """
    List all available LLM providers with their capabilities
    """
    try:
        providers_info = []
        
        # OpenAI GPT-4
        providers_info.append(LLMProviderInfo(
            provider="openai-gpt4",
            model_name="gpt-4-turbo-preview",
            cost_per_1k_tokens=0.03,
            max_tokens=4096,
            temperature=0.7,
            description="OpenAI's flagship model with excellent reasoning and creativity",
            strengths=["Complex reasoning", "Creative writing", "Code review", "Analysis"],
            use_cases=["Research", "Complex problem solving", "Content creation", "Code analysis"]
        ))
        
        # OpenAI GPT-4o
        providers_info.append(LLMProviderInfo(
            provider="openai-gpt4o",
            model_name="gpt-4o",
            cost_per_1k_tokens=0.005,
            max_tokens=4096,
            temperature=0.7,
            description="OpenAI's optimized model balancing performance and cost-effectiveness",
            strengths=["Cost efficiency", "Fast inference", "Multimodal", "Balanced performance"],
            use_cases=["General tasks", "Cost-sensitive applications", "Image analysis", "Automation"]
        ))
        
        # OpenAI GPT-4.1
        providers_info.append(LLMProviderInfo(
            provider="openai-gpt4.1",
            model_name="gpt-4.1-preview",
            cost_per_1k_tokens=0.035,
            max_tokens=8192,
            temperature=0.7,
            description="Next-generation GPT-4 with enhanced capabilities and larger context",
            strengths=["Enhanced reasoning", "Larger context", "Improved accuracy", "Advanced analysis"],
            use_cases=["Long document analysis", "Complex reasoning", "Advanced research", "Technical writing"]
        ))
        
        # OpenAI GPT-o3
        providers_info.append(LLMProviderInfo(
            provider="openai-gpt-o3",
            model_name="gpt-o3",
            cost_per_1k_tokens=0.06,
            max_tokens=4096,
            temperature=0.7,
            description="OpenAI's premium reasoning model optimized for complex problem solving",
            strengths=["Advanced reasoning", "Mathematical thinking", "Logic problems", "Scientific analysis"],
            use_cases=["Mathematical proofs", "Scientific research", "Complex reasoning", "Academic analysis"]
        ))
        
        # OpenAI GPT-3.5
        providers_info.append(LLMProviderInfo(
            provider="openai-gpt3.5",
            model_name="gpt-3.5-turbo",
            cost_per_1k_tokens=0.002,
            max_tokens=4096,
            temperature=0.7,
            description="Fast and cost-effective model for general tasks",
            strengths=["Speed", "Cost efficiency", "General conversation", "Simple tasks"],
            use_cases=["Chat", "Q&A", "Simple coding", "Text processing"]
        ))
        
        # DeepSeek V2
        providers_info.append(LLMProviderInfo(
            provider="deepseek-v2",
            model_name="deepseek-chat",
            cost_per_1k_tokens=0.0014,
            max_tokens=4096,
            temperature=0.7,
            description="DeepSeek's general-purpose model with strong performance",
            strengths=["Cost efficiency", "Reasoning", "Multilingual", "Fast inference"],
            use_cases=["General chat", "Analysis", "Translation", "Research"]
        ))
        
        # DeepSeek V3
        providers_info.append(LLMProviderInfo(
            provider="deepseek-v3",
            model_name="deepseek-v3",
            cost_per_1k_tokens=0.0027,
            max_tokens=8192,
            temperature=0.7,
            description="DeepSeek's latest model with enhanced capabilities and larger context",
            strengths=["Enhanced reasoning", "Larger context", "Improved performance", "Multi-domain expertise"],
            use_cases=["Complex analysis", "Long document processing", "Advanced reasoning", "Research"]
        ))
        
        # DeepSeek R1
        providers_info.append(LLMProviderInfo(
            provider="deepseek-r1",
            model_name="deepseek-r1",
            cost_per_1k_tokens=0.008,
            max_tokens=4096,
            temperature=0.7,
            description="DeepSeek's reasoning-specialized model for complex problem solving",
            strengths=["Advanced reasoning", "Mathematical thinking", "Logical deduction", "Problem solving"],
            use_cases=["Mathematical problems", "Logic puzzles", "Scientific reasoning", "Complex analysis"]
        ))
        
        # DeepSeek Coder
        providers_info.append(LLMProviderInfo(
            provider="deepseek-coder",
            model_name="deepseek-coder",
            cost_per_1k_tokens=0.0014,
            max_tokens=4096,
            temperature=0.7,
            description="Specialized model for coding and technical tasks",
            strengths=["Code generation", "Debugging", "Technical documentation", "Architecture"],
            use_cases=["Software development", "Code review", "Technical writing", "Debugging"]
        ))
        
        # Grok Beta
        providers_info.append(LLMProviderInfo(
            provider="grok-beta",
            model_name="grok-beta",
            cost_per_1k_tokens=0.005,
            max_tokens=4096,
            temperature=0.7,
            description="X.AI's Grok model with real-time information access",
            strengths=["Real-time data", "Wit and humor", "Current events", "Conversational"],
            use_cases=["Current events", "Real-time analysis", "Social media", "News"]
        ))
        
        logger.info(f"Retrieved {len(providers_info)} LLM providers")
        
        return ApiResponse(
            success=True,
            data=providers_info,
            metadata={
                "total_providers": len(providers_info),
                "categories": ["general", "coding", "creative", "real-time"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing LLM providers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while listing providers"
        )

@router.post("/select", response_model=ApiResponse[LLMSelectionResponse])
async def select_optimal_provider(
    request: LLMSelectionRequest
) -> ApiResponse[LLMSelectionResponse]:
    """
    Select optimal LLM provider based on task requirements and budget
    """
    try:
        logger.info(f"Selecting optimal provider for task: {request.task_type}")
        
        # Get optimal provider recommendation
        recommended_provider = llm_provider_service.select_optimal_provider(
            task_type=request.task_type,
            budget_limit=request.budget_limit,
            token_estimate=request.token_estimate
        )
        
        # Calculate cost estimate
        cost_estimate = llm_provider_service.estimate_cost(
            provider=recommended_provider,
            token_count=request.token_estimate
        )
        
        # Generate reasoning
        provider_info = llm_provider_service.get_provider_info(recommended_provider)
        reasoning = f"Selected {recommended_provider.value} based on task type '{request.task_type}' and budget constraint of ${request.budget_limit}. This provider offers the best balance of capability and cost for your requirements."
        
        # Get alternative options
        all_providers = llm_provider_service.list_providers()
        alternatives = []
        
        for provider_data in all_providers:
            if provider_data["provider"] != recommended_provider.value:
                alt_cost = llm_provider_service.estimate_cost(
                    provider=LLMProvider(provider_data["provider"]),
                    token_count=request.token_estimate
                )
                alternatives.append({
                    "provider": provider_data["provider"],
                    "cost_estimate": alt_cost,
                    "suitable": alt_cost <= request.budget_limit
                })
        
        # Sort alternatives by cost
        alternatives.sort(key=lambda x: x["cost_estimate"])
        
        response = LLMSelectionResponse(
            recommended_provider=recommended_provider.value,
            reasoning=reasoning,
            cost_estimate=cost_estimate,
            alternatives=alternatives[:3]  # Top 3 alternatives
        )
        
        logger.info(f"Recommended provider: {recommended_provider.value} (${cost_estimate:.4f})")
        
        return ApiResponse(
            success=True,
            data=response,
            metadata={
                "task_type": request.task_type,
                "budget_limit": request.budget_limit,
                "token_estimate": request.token_estimate,
                "selection_criteria": {
                    "cost_optimization": cost_estimate <= request.budget_limit,
                    "task_suitability": True
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error selecting optimal provider: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while selecting provider"
        )

@router.post("/configure", response_model=ApiResponse[Dict[str, Any]])
async def configure_provider(
    request: LLMConfigRequest
) -> ApiResponse[Dict[str, Any]]:
    """
    Configure parameters for a specific LLM provider
    """
    try:
        logger.info(f"Configuring provider: {request.provider}")
        
        # Validate provider exists
        try:
            provider_enum = LLMProvider(request.provider)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.provider}"
            )
        
        # Create configuration
        config = LLMConfig(
            provider=provider_enum,
            model_name=llm_provider_service.providers[provider_enum].model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )
        
        # Apply configuration
        llm_provider_service.configure_provider(provider_enum, config)
        
        response_data = {
            "provider": request.provider,
            "configuration": config.dict(),
            "status": "configured"
        }
        
        logger.info(f"Provider {request.provider} configured successfully")
        
        return ApiResponse(
            success=True,
            data=response_data,
            metadata={
                "configured_at": "now",
                "parameters_updated": [
                    "temperature", "max_tokens", "top_p", 
                    "frequency_penalty", "presence_penalty"
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring provider: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while configuring provider"
        )

@router.get("/cost-estimate", response_model=ApiResponse[Dict[str, float]])
async def estimate_costs(
    provider: str,
    token_count: int
) -> ApiResponse[Dict[str, float]]:
    """
    Estimate costs for different LLM providers
    """
    try:
        if provider:
            # Estimate for specific provider
            try:
                provider_enum = LLMProvider(provider)
                cost = llm_provider_service.estimate_cost(provider_enum, token_count)
                result = {provider: cost}
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid provider: {provider}"
                )
        else:
            # Estimate for all providers
            result = {}
            for provider_enum in LLMProvider:
                cost = llm_provider_service.estimate_cost(provider_enum, token_count)
                result[provider_enum.value] = cost
        
        logger.info(f"Cost estimates calculated for {token_count} tokens")
        
        return ApiResponse(
            success=True,
            data=result,
            metadata={
                "token_count": token_count,
                "currency": "USD",
                "rate_basis": "per_1k_tokens"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating costs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while estimating costs"
        )

@router.post("/store-api-key", response_model=ApiResponse[Dict[str, Any]])
async def store_user_api_key(
    request: APIKeyRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, Any]]:
    """
    Store a user's API key for a specific LLM provider
    """
    try:
        logger.info(f"Storing API key for session {request.session_id}, provider {request.provider}")
        
        # Validate provider exists
        try:
            provider_enum = LLMProvider(request.provider)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.provider}"
            )
        
        # Store the encrypted API key
        user_key = await api_key_manager.store_api_key(
            db=db,
            session_id=request.session_id,
            provider=request.provider,
            api_key=request.api_key
        )
        
        # Validate the key asynchronously
        is_valid = await api_key_manager.validate_api_key(
            db=db,
            session_id=request.session_id,
            provider=request.provider
        )
        
        response_data = {
            "provider": request.provider,
            "stored": True,
            "is_valid": is_valid,
            "expires_at": user_key.expires_at.isoformat()
        }
        
        logger.info(f"API key stored and validated for {request.provider}: {is_valid}")
        
        return ApiResponse(
            success=True,
            data=response_data,
            metadata={
                "session_id": request.session_id,
                "validation_performed": True
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing API key: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while storing API key"
        )

@router.post("/validate-api-key", response_model=ApiResponse[Dict[str, Any]])
async def validate_user_api_key(
    request: APIKeyValidationRequest,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, Any]]:
    """
    Validate a stored API key for a specific provider
    """
    try:
        logger.info(f"Validating API key for session {request.session_id}, provider {request.provider}")
        
        # Validate the key
        is_valid = await api_key_manager.validate_api_key(
            db=db,
            session_id=request.session_id,
            provider=request.provider
        )
        
        response_data = {
            "provider": request.provider,
            "is_valid": is_valid,
            "validated_at": "now"
        }
        
        logger.info(f"API key validation result for {request.provider}: {is_valid}")
        
        return ApiResponse(
            success=True,
            data=response_data,
            metadata={
                "session_id": request.session_id,
                "validation_timestamp": "now"
            }
        )
        
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while validating API key"
        )

@router.get("/user-api-keys/{session_id}", response_model=ApiResponse[List[APIKeyResponse]])
async def list_user_api_keys(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[List[APIKeyResponse]]:
    """
    List all API keys for a user session
    """
    try:
        logger.info(f"Listing API keys for session {session_id}")
        
        # Get user's API keys
        user_keys = await api_key_manager.list_user_keys(db, session_id)
        
        # Convert to response format
        api_key_responses = []
        for key_info in user_keys:
            api_key_responses.append(APIKeyResponse(
                provider=key_info["provider"],
                is_valid=key_info["is_valid"],
                last_validated=key_info["last_validated"].isoformat() if key_info["last_validated"] else None,
                expires_at=key_info["expires_at"].isoformat(),
                needs_validation=key_info["needs_validation"]
            ))
        
        logger.info(f"Found {len(api_key_responses)} API keys for session {session_id}")
        
        return ApiResponse(
            success=True,
            data=api_key_responses,
            metadata={
                "session_id": session_id,
                "total_keys": len(api_key_responses)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing user API keys: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while listing API keys"
        )

@router.delete("/user-api-key", response_model=ApiResponse[Dict[str, Any]])
async def remove_user_api_key(
    session_id: str,
    provider: str,
    db: AsyncSession = Depends(get_db)
) -> ApiResponse[Dict[str, Any]]:
    """
    Remove a user's API key for a specific provider
    """
    try:
        logger.info(f"Removing API key for session {session_id}, provider {provider}")
        
        # Remove the API key
        removed = await api_key_manager.remove_api_key(
            db=db,
            session_id=session_id,
            provider=provider
        )
        
        response_data = {
            "provider": provider,
            "removed": removed
        }
        
        logger.info(f"API key removal result for {provider}: {removed}")
        
        return ApiResponse(
            success=True,
            data=response_data,
            metadata={
                "session_id": session_id,
                "operation": "remove"
            }
        )
        
    except Exception as e:
        logger.error(f"Error removing API key: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while removing API key"
        )