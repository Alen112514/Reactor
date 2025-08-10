'use client'

import { useState, useEffect } from 'react'
import { X, Key, Settings, Save, Eye, EyeOff } from 'lucide-react'

interface LLMProvider {
  id: string
  name: string
  description: string
  defaultModel: string
}

interface UserPreferences {
  preferred_provider: string
  temperature: number
  max_tokens: number
  cost_limit: number
  has_api_key: boolean
}

interface ApiKey {
  id: string
  provider: string
  masked_key: string
  created_at: string
}

interface SettingsPanelProps {
  isOpen: boolean
  onClose: () => void
  sessionId: string
  onSettingsUpdated: () => void
}

const SUPPORTED_PROVIDERS: LLMProvider[] = [
  {
    id: 'openai-gpt4o',
    name: 'OpenAI GPT-4o',
    description: 'Latest GPT-4 Omni model with vision and reasoning capabilities',
    defaultModel: 'gpt-4o'
  },
  {
    id: 'openai-gpt4o-mini',
    name: 'OpenAI GPT-4o Mini',
    description: 'Faster, more affordable GPT-4 model',
    defaultModel: 'gpt-4o-mini'
  },
  {
    id: 'openai-gpt4',
    name: 'OpenAI GPT-4',
    description: 'Previous generation GPT-4 model',
    defaultModel: 'gpt-4'
  },
  {
    id: 'deepseek-v2',
    name: 'DeepSeek Chat',
    description: 'DeepSeek conversational AI model',
    defaultModel: 'deepseek-chat'
  },
  {
    id: 'grok-beta',
    name: 'Grok Beta',
    description: 'xAI Grok model (beta version)',
    defaultModel: 'grok-beta'
  }
]

export default function SettingsPanel({ isOpen, onClose, sessionId, onSettingsUpdated }: SettingsPanelProps) {
  const [preferences, setPreferences] = useState<UserPreferences>({
    preferred_provider: 'openai-gpt4o',
    temperature: 0.7,
    max_tokens: 4096,
    cost_limit: 10.0,
    has_api_key: false
  })
  
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  const [newApiKey, setNewApiKey] = useState('')
  const [selectedProvider, setSelectedProvider] = useState('openai-gpt4o')
  const [showApiKey, setShowApiKey] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    if (isOpen) {
      loadPreferences()
      loadApiKeys()
    }
  }, [isOpen, sessionId])

  // Sync selectedProvider with preferred_provider when preferences load or change
  useEffect(() => {
    setSelectedProvider(preferences.preferred_provider)
  }, [preferences.preferred_provider])

  const loadPreferences = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/user/llm-preferences?session_id=${sessionId}`)
      const result = await response.json()
      
      if (result.success) {
        setPreferences(result.data)
        setSelectedProvider(result.data.preferred_provider)
      }
    } catch (error) {
      console.error('Error loading preferences:', error)
    }
  }

  const loadApiKeys = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/llm-providers/user-api-keys/${sessionId}`)
      const result = await response.json()
      
      if (result.success) {
        // Convert backend format to frontend format
        const formattedKeys = result.data.map((key: any, index: number) => ({
          id: key.provider, // Use provider as ID since there's one per provider
          provider: key.provider,
          masked_key: key.masked_key || '••••••••••••',
          created_at: key.last_validated || new Date().toISOString(),
          is_valid: key.is_valid
        }))
        setApiKeys(formattedKeys)
      }
    } catch (error) {
      console.error('Error loading API keys:', error)
    }
  }

  const savePreferences = async () => {
    setIsSaving(true)
    try {
      const response = await fetch('http://localhost:8000/api/v1/user/llm-preferences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          preferred_provider: preferences.preferred_provider,
          temperature: preferences.temperature,
          max_tokens: preferences.max_tokens,
          cost_limit: preferences.cost_limit
        }),
      })

      const result = await response.json()
      
      if (result.success) {
        onSettingsUpdated()
        onClose() // Auto-close panel after successful save
      } else {
        alert('Failed to save preferences: ' + (result.error || 'Unknown error'))
      }
    } catch (error) {
      console.error('Error saving preferences:', error)
      alert('Failed to save preferences')
    } finally {
      setIsSaving(false)
    }
  }

  const addApiKey = async () => {
    const trimmedKey = newApiKey.trim()
    
    if (!trimmedKey) {
      alert('Please enter an API key')
      return
    }

    if (trimmedKey.length < 10) {
      alert('API key must be at least 10 characters long')
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/v1/llm-providers/store-api-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          provider: selectedProvider,
          api_key: trimmedKey
        }),
      })

      const result = await response.json()
      
      if (response.ok && result.success) {
        setNewApiKey('')
        await loadApiKeys()
        onSettingsUpdated()
        
        // Show success message
        if (result.data.is_valid === false) {
          alert('API key added successfully, but validation failed. Please check if the key is correct.')
        } else {
          alert('API key added successfully!')
        }
      } else {
        // Handle specific error cases
        if (result.detail && Array.isArray(result.detail)) {
          const errorMsg = result.detail.map((err: any) => err.msg).join(', ')
          alert('Validation error: ' + errorMsg)
        } else {
          alert('Failed to add API key: ' + (result.error || result.detail || 'Unknown error'))
        }
      }
    } catch (error) {
      console.error('Error adding API key:', error)
      alert('Network error: Failed to connect to server')
    } finally {
      setIsLoading(false)
    }
  }

  const deleteApiKey = async (provider: string) => {
    if (!confirm('Are you sure you want to delete this API key?')) return

    try {
      const response = await fetch(`http://localhost:8000/api/v1/llm-providers/user-api-key?session_id=${sessionId}&provider=${provider}`, {
        method: 'DELETE'
      })

      const result = await response.json()
      
      if (result.success) {
        await loadApiKeys()
        onSettingsUpdated()
      } else {
        alert('Failed to delete API key: ' + (result.error || 'Unknown error'))
      }
    } catch (error) {
      console.error('Error deleting API key:', error)
      alert('Failed to delete API key')
    }
  }

  const getProviderName = (providerId: string) => {
    return SUPPORTED_PROVIDERS.find(p => p.id === providerId)?.name || providerId
  }

  // Component for Tavily API Key Management
  function TavilyApiKeySection({ 
    apiKeys, 
    onApiKeysChange, 
    sessionId 
  }: { 
    apiKeys: ApiKey[], 
    onApiKeysChange: (keys: ApiKey[]) => void,
    sessionId: string 
  }) {
    const [tavilyApiKey, setTavilyApiKey] = useState('')
    const [showTavilyKey, setShowTavilyKey] = useState(false)
    const [isAddingTavily, setIsAddingTavily] = useState(false)
    
    const tavilyKey = apiKeys.find(key => key.provider === 'tavily')
    
    const addTavilyApiKey = async () => {
      if (!tavilyApiKey.trim() || tavilyApiKey.trim().length < 10) return
      
      setIsAddingTavily(true)
      try {
        const response = await fetch('http://localhost:8000/api/v1/llm-providers/store-api-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            provider: 'tavily',
            api_key: tavilyApiKey.trim()
          })
        })
        
        if (response.ok) {
          setTavilyApiKey('')
          // Reload API keys
          const keysResponse = await fetch(`http://localhost:8000/api/v1/llm-providers/api-keys?session_id=${sessionId}`)
          if (keysResponse.ok) {
            const keysResult = await keysResponse.json()
            onApiKeysChange(keysResult.data || [])
          }
        } else {
          alert('Failed to add Tavily API key')
        }
      } catch (error) {
        console.error('Error adding Tavily API key:', error)
        alert('Failed to add Tavily API key')
      } finally {
        setIsAddingTavily(false)
      }
    }
    
    const deleteTavilyApiKey = async () => {
      if (!confirm('Are you sure you want to delete your Tavily API key?')) return
      
      try {
        const response = await fetch(`http://localhost:8000/api/v1/llm-providers/api-keys/tavily?session_id=${sessionId}`, {
          method: 'DELETE'
        })
        
        if (response.ok) {
          // Reload API keys
          const keysResponse = await fetch(`http://localhost:8000/api/v1/llm-providers/api-keys?session_id=${sessionId}`)
          if (keysResponse.ok) {
            const keysResult = await keysResponse.json()
            onApiKeysChange(keysResult.data || [])
          }
        } else {
          alert('Failed to delete Tavily API key')
        }
      } catch (error) {
        console.error('Error deleting Tavily API key:', error)
        alert('Failed to delete Tavily API key')
      }
    }
    
    return (
      <div>
        {tavilyKey ? (
          <div className="flex items-center justify-between p-3 bg-green-900/20 border border-green-500/30 rounded">
            <div>
              <div className="font-medium text-green-300 flex items-center">
                <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                Tavily API Key Configured
              </div>
              <div className="text-sm text-gray-300">Key: {tavilyKey.masked_key}</div>
              <div className="text-xs text-gray-400">
                Added: {new Date(tavilyKey.created_at).toLocaleDateString()}
              </div>
            </div>
            <button
              onClick={deleteTavilyApiKey}
              className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
            >
              Delete
            </button>
          </div>
        ) : (
          <div>
            <div className="flex space-x-2 mb-2">
              <div className="flex-1 relative">
                <input
                  type={showTavilyKey ? "text" : "password"}
                  placeholder="Enter Tavily API key (min 10 characters)..."
                  value={tavilyApiKey}
                  onChange={(e) => setTavilyApiKey(e.target.value)}
                  className={`w-full p-2 bg-slate-600 rounded border pr-10 focus:outline-none ${
                    tavilyApiKey.trim().length > 0 && tavilyApiKey.trim().length < 10
                      ? 'border-red-500 focus:border-red-500'
                      : 'border-slate-500 focus:border-blue-500'
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowTavilyKey(!showTavilyKey)}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showTavilyKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              
              <button
                onClick={addTavilyApiKey}
                disabled={isAddingTavily || !tavilyApiKey.trim() || tavilyApiKey.trim().length < 10}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600"
              >
                {isAddingTavily ? 'Adding...' : 'Add Key'}
              </button>
            </div>
            
            {tavilyApiKey.trim().length > 0 && tavilyApiKey.trim().length < 10 && (
              <div className="text-sm text-red-400 mb-2">
                API key must be at least 10 characters long (currently {tavilyApiKey.trim().length})
              </div>
            )}
            
            <div className="text-xs text-gray-400">
              <p className="mb-1">Get your free API key at <a href="https://app.tavily.com" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">app.tavily.com</a></p>
              <p>🔓 Optional: Web search works without API key but with basic functionality</p>
            </div>
          </div>
        )}
      </div>
    )
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-slate-800 text-white rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center space-x-2">
            <Settings className="w-6 h-6" />
            <h2 className="text-2xl font-bold">Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-8">
          {/* LLM Provider Selection */}
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Settings className="w-5 h-5 mr-2" />
              LLM Provider & Model
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {SUPPORTED_PROVIDERS.map((provider) => (
                <div
                  key={provider.id}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                    preferences.preferred_provider === provider.id
                      ? 'border-blue-500 bg-blue-900/20'
                      : 'border-slate-600 bg-slate-700 hover:border-slate-500'
                  }`}
                  onClick={() => {
                    setPreferences({...preferences, preferred_provider: provider.id})
                    setSelectedProvider(provider.id)
                  }}
                >
                  <div className="font-semibold">{provider.name}</div>
                  <div className="text-sm text-gray-300 mt-1">{provider.description}</div>
                  <div className="text-xs text-gray-400 mt-2">Model: {provider.defaultModel}</div>
                  {apiKeys.find(key => key.provider === provider.id) && (
                    <div className="text-xs text-green-400 mt-1">✓ API Key configured</div>
                  )}
                </div>
              ))}
            </div>

            {/* Advanced Settings */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Temperature</label>
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.1"
                  value={preferences.temperature}
                  onChange={(e) => setPreferences({...preferences, temperature: parseFloat(e.target.value)})}
                  className="w-full p-2 bg-slate-700 rounded border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
                <p className="text-xs text-gray-400 mt-1">0 = deterministic, 2 = very creative</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Max Tokens</label>
                <input
                  type="number"
                  min="1"
                  max="32768"
                  value={preferences.max_tokens}
                  onChange={(e) => setPreferences({...preferences, max_tokens: parseInt(e.target.value)})}
                  className="w-full p-2 bg-slate-700 rounded border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
                <p className="text-xs text-gray-400 mt-1">Maximum response length</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Cost Limit ($)</label>
                <input
                  type="number"
                  min="0.01"
                  step="0.01"
                  value={preferences.cost_limit}
                  onChange={(e) => setPreferences({...preferences, cost_limit: parseFloat(e.target.value)})}
                  className="w-full p-2 bg-slate-700 rounded border border-slate-600 focus:border-blue-500 focus:outline-none"
                />
                <p className="text-xs text-gray-400 mt-1">Per-query spending limit</p>
              </div>
            </div>
          </div>

          {/* API Key Management */}
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Key className="w-5 h-5 mr-2" />
              API Keys
            </h3>

            {/* Add New API Key */}
            <div className="bg-slate-700 p-4 rounded-lg mb-4">
              <h4 className="font-medium mb-3">
                Add API Key for {getProviderName(selectedProvider)}
              </h4>
              <div className="flex space-x-2 mb-2">
                <div className="flex-1 relative">
                  <input
                    type={showApiKey ? "text" : "password"}
                    placeholder={`Enter ${getProviderName(selectedProvider)} API key (min 10 characters)...`}
                    value={newApiKey}
                    onChange={(e) => setNewApiKey(e.target.value)}
                    className={`w-full p-2 bg-slate-600 rounded border pr-10 focus:outline-none ${
                      newApiKey.trim().length > 0 && newApiKey.trim().length < 10
                        ? 'border-red-500 focus:border-red-500'
                        : 'border-slate-500 focus:border-blue-500'
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                  >
                    {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                
                <button
                  onClick={addApiKey}
                  disabled={isLoading || !newApiKey.trim() || newApiKey.trim().length < 10}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600"
                >
                  {isLoading ? 'Adding...' : 'Add Key'}
                </button>
              </div>
              
              {/* Validation feedback */}
              {newApiKey.trim().length > 0 && newApiKey.trim().length < 10 && (
                <div className="text-sm text-red-400 mb-2">
                  API key must be at least 10 characters long (currently {newApiKey.trim().length})
                </div>
              )}
              <p className="text-xs text-gray-400">
                API keys are encrypted and stored securely. Never share your keys.
              </p>
              
              {/* Show existing key status for selected provider */}
              {apiKeys.find(key => key.provider === selectedProvider) ? (
                <div className="mt-2 text-sm text-green-400 flex items-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                  API key already configured for {getProviderName(selectedProvider)}
                </div>
              ) : (
                <div className="mt-2 text-sm text-yellow-400 flex items-center">
                  <span className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></span>
                  No API key configured for {getProviderName(selectedProvider)}
                </div>
              )}
            </div>

            {/* Existing API Keys */}
            <div>
              <h5 className="font-medium mb-3 text-gray-300">Configured API Keys</h5>
              <div className="space-y-2">
                {apiKeys.length === 0 ? (
                  <div className="text-center text-gray-400 py-6">
                    <Key className="w-10 h-10 mx-auto mb-2 opacity-50" />
                    <p>No API keys configured</p>
                    <p className="text-sm">Select a provider above and add an API key to get started</p>
                  </div>
                ) : (
                  <>
                    {/* Show currently selected provider's key first if it exists */}
                    {apiKeys
                      .filter(key => key.provider === selectedProvider)
                      .map((key) => (
                        <div key={key.id} className="flex items-center justify-between p-3 bg-blue-900/30 border border-blue-500/30 rounded">
                          <div>
                            <div className="font-medium text-blue-300">{getProviderName(key.provider)} (Selected)</div>
                            <div className="text-sm text-gray-300">Key: {key.masked_key}</div>
                            <div className="text-xs text-gray-400">
                              Added: {new Date(key.created_at).toLocaleDateString()}
                            </div>
                          </div>
                          <button
                            onClick={() => deleteApiKey(key.provider)}
                            className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
                          >
                            Delete
                          </button>
                        </div>
                      ))
                    }
                    
                    {/* Show other providers' keys */}
                    {apiKeys
                      .filter(key => key.provider !== selectedProvider)
                      .map((key) => (
                        <div key={key.id} className="flex items-center justify-between p-3 bg-slate-700 rounded">
                          <div>
                            <div className="font-medium">{getProviderName(key.provider)}</div>
                            <div className="text-sm text-gray-300">Key: {key.masked_key}</div>
                            <div className="text-xs text-gray-400">
                              Added: {new Date(key.created_at).toLocaleDateString()}
                            </div>
                          </div>
                          <button
                            onClick={() => deleteApiKey(key.provider)}
                            className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
                          >
                            Delete
                          </button>
                        </div>
                      ))
                    }
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Tavily Web Search API Key (Optional) */}
          <div className="mt-6 pt-6 border-t border-slate-600">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <span className="w-5 h-5 mr-2 text-lg">🔍</span>
              Tavily Web Search (Optional)
            </h3>
            
            <div className="bg-slate-700/50 p-4 rounded-lg">
              <div className="mb-4">
                <h4 className="font-medium mb-2">Web Search & Content Extraction</h4>
                <p className="text-sm text-gray-400 mb-3">
                  Tavily provides AI-powered web search and content extraction. Configure an API key to enable web browsing tools.
                </p>
                <div className="text-xs text-gray-500">
                  • Real-time web search • Content extraction • Website viewing
                  <br />
                  • Optional: Works without API key but with limited functionality
                </div>
              </div>
              
              <TavilyApiKeySection 
                apiKeys={apiKeys} 
                onApiKeysChange={setApiKeys}
                sessionId={sessionId}
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-slate-700">
          <div className="text-sm text-gray-400">
            Session ID: {sessionId.slice(0, 8)}...
          </div>
          <div className="flex space-x-2">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-700 text-white rounded hover:bg-slate-600"
            >
              Cancel
            </button>
            <button
              onClick={savePreferences}
              disabled={isSaving}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
            >
              <Save className="w-4 h-4" />
              <span>{isSaving ? 'Saving...' : 'Save Settings'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}