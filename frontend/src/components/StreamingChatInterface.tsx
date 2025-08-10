'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Send, Loader2, Settings, History, Monitor, Activity, Zap } from 'lucide-react'
import SettingsPanel from './SettingsPanel'
import SplitScreenLayout from './SplitScreenLayout'
import BrowserShell from './browser/BrowserShell'
import MCPToolsPanel from './MCPToolsPanel'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool' | 'workflow'
  content: string
  timestamp: string
  tools_used?: string[]
  is_streaming?: boolean
  tool_status?: 'starting' | 'running' | 'completed' | 'error'
  tool_name?: string
  workflow_step?: string
  workflow_status?: 'in_progress' | 'completed' | 'error'
  workflow_metadata?: any
}

interface StreamingChatInterfaceProps {
  sessionId: string
}

export default function StreamingChatInterface({ sessionId }: StreamingChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSetupComplete, setIsSetupComplete] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [splitScreenMode, setSplitScreenMode] = useState(false)
  const [currentWebsiteUrl, setCurrentWebsiteUrl] = useState('')
  const [splitPercentage, setSplitPercentage] = useState(45)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  const [streamingStats, setStreamingStats] = useState({ messagesReceived: 0, toolsExecuted: 0, workflowSteps: 0 })
  const [screenshotData, setScreenshotData] = useState<{
    base64Data: string
    url: string
    title?: string
    timestamp?: number
  } | null>(null)
  const [processedScreenshots, setProcessedScreenshots] = useState<Set<string>>(new Set())
  const [recentMessages, setRecentMessages] = useState<Set<string>>(new Set())

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const messageStreamWs = useRef<WebSocket | null>(null)
  const streamingMessages = useRef<Map<string, Message>>(new Map())

  // Add state change logging
  useEffect(() => {
    console.log('🔧 STATE DEBUG: splitScreenMode changed to:', splitScreenMode)
  }, [splitScreenMode])

  useEffect(() => {
    console.log('🔧 STATE DEBUG: currentWebsiteUrl changed to:', currentWebsiteUrl)
    
    // Clear screenshot tracking when URL changes (new page navigation)
    if (currentWebsiteUrl) {
      setProcessedScreenshots(new Set())
      console.log('🔄 RESET: Cleared screenshot tracking for new URL:', currentWebsiteUrl)
    }
  }, [currentWebsiteUrl])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Initialize services
    checkSessionStatus()
    // Note: loadConversationHistory() removed - no automatic history loading on refresh
    connectMessageStream()
    
    return () => {
      disconnectMessageStream()
    }
  }, [sessionId])

  const connectMessageStream = useCallback(() => {
    if (messageStreamWs.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected, skipping reconnection')
      return
    }

    console.log('Attempting to connect to message streaming WebSocket...')
    setConnectionStatus('connecting')
    const wsUrl = `ws://localhost:8000/api/v1/ws/message-stream?user_id=${sessionId}&session_id=${sessionId}`
    console.log('WebSocket URL:', wsUrl)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('✅ Message streaming WebSocket connected successfully')
      setConnectionStatus('connected')
    }

    ws.onmessage = (event) => {
      console.log('📨 WebSocket message received:', event.data)
      try {
        const message = JSON.parse(event.data)
        console.log('📄 Parsed message:', message)
        
        // Enhanced WebSocket debugging
        console.log('🔧 WEBSOCKET DEBUG: Message type:', message.type)
        console.log('🔧 WEBSOCKET DEBUG: Complete message keys:', Object.keys(message))
        
        if (message.type === 'browser_action') {
          console.log('🎯 WEBSOCKET DEBUG: BROWSER_ACTION MESSAGE DETECTED!')
          console.log('🎯 WEBSOCKET DEBUG: browser_action details:', {
            action_type: message.action_type,
            website_url: message.website_url,
            enable_split_screen: message.enable_split_screen,
            tool_name: message.tool_name
          })
        }
        
        if (message.type === 'tool_execution_update') {
          console.log('🔧 WEBSOCKET DEBUG: tool_execution_update:', {
            tool_name: message.tool_name,
            status: message.status,
            result_keys: message.result ? Object.keys(message.result) : 'no result'
          })
        }
        
        handleStreamingMessage(message)
      } catch (error) {
        console.error('❌ Failed to parse streaming message:', error, 'Raw data:', event.data)
      }
    }

    ws.onclose = () => {
      console.log('Message streaming disconnected')
      setConnectionStatus('disconnected')
      
      // Auto-reconnect after delay
      setTimeout(() => {
        if (!messageStreamWs.current || messageStreamWs.current.readyState === WebSocket.CLOSED) {
          connectMessageStream()
        }
      }, 3000)
    }

    ws.onerror = (error) => {
      console.error('Message streaming error:', error)
      setConnectionStatus('disconnected')
    }

    messageStreamWs.current = ws
  }, [sessionId])

  const disconnectMessageStream = useCallback(() => {
    if (messageStreamWs.current) {
      messageStreamWs.current.close()
      messageStreamWs.current = null
    }
    setConnectionStatus('disconnected')
  }, [])

  const handleStreamingMessage = (streamingMessage: any) => {
    console.log('🔍 Handling streaming message type:', streamingMessage.type)
    const { type, message_id, content, chunk, tool_name, status, progress_message } = streamingMessage

    switch (type) {
      case 'connection_established':
        console.log('✅ Message streaming connection established:', streamingMessage.connection_id)
        break

      case 'message_started':
        // Start new streaming message
        if (streamingMessage.message_type === 'assistant_message') {
          const newMessage: Message = {
            id: message_id,
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString(),
            is_streaming: true
          }
          streamingMessages.current.set(message_id, newMessage)
          setMessages(prev => [...prev, newMessage])
        } else if (streamingMessage.message_type === 'tool_execution_start') {
          const toolMessage: Message = {
            id: message_id,
            role: 'tool',
            content: `Starting ${tool_name || 'tool'}...`,
            timestamp: new Date().toISOString(),
            tool_status: 'starting',
            tool_name: tool_name,
            is_streaming: true
          }
          streamingMessages.current.set(message_id, toolMessage)
          setMessages(prev => [...prev, toolMessage])
          setStreamingStats(prev => ({ ...prev, toolsExecuted: prev.toolsExecuted + 1 }))
        }
        break

      case 'message_chunk':
        // Update streaming message with new chunk
        const existingMessage = streamingMessages.current.get(message_id)
        if (existingMessage) {
          const updatedMessage = {
            ...existingMessage,
            content: content || (existingMessage.content + chunk)
          }
          streamingMessages.current.set(message_id, updatedMessage)
          
          setMessages(prev => prev.map(msg => 
            msg.id === message_id ? updatedMessage : msg
          ))

          // Extract screenshot data from assistant messages (with deduplication)
          if (updatedMessage.role === 'assistant' && updatedMessage.content) {
            const screenshotMatch = updatedMessage.content.match(/!\[.*?\]\(data:image\/[^)]+\)/g)
            if (screenshotMatch) {
              const screenshotUrl = currentWebsiteUrl || 'unknown'
              const screenshotKey = `${screenshotUrl}_${Date.now()}`
              
              // Check if we've already processed a screenshot for this URL recently
              if (!processedScreenshots.has(screenshotUrl)) {
                console.log('📸 SCREENSHOT: Found new screenshot data in assistant message')
                const base64Match = screenshotMatch[0].match(/data:image\/[^)]+/)
                if (base64Match) {
                  const screenshotData = {
                    base64Data: base64Match[0],
                    url: screenshotUrl,
                    title: 'AI Generated Screenshot',
                    timestamp: Date.now()
                  }
                  console.log('📸 SCREENSHOT: Setting screenshot data for URL:', screenshotUrl)
                  setScreenshotData(screenshotData)
                  
                  // Mark this URL as processed
                  setProcessedScreenshots(prev => new Set([...prev, screenshotUrl]))
                  
                  // Clear screenshot data after a delay
                  setTimeout(() => {
                    setScreenshotData(null)
                  }, 1000)
                }
              } else {
                console.log('📸 SCREENSHOT: Skipping duplicate screenshot for URL:', screenshotUrl)
              }
            }
          }
        }
        setStreamingStats(prev => ({ ...prev, messagesReceived: prev.messagesReceived + 1 }))
        break

      case 'message_complete':
        // Finalize streaming message
        const completedMessage = streamingMessages.current.get(message_id)
        if (completedMessage) {
          const finalMessage = {
            ...completedMessage,
            content: content || completedMessage.content,
            is_streaming: false
          }
          streamingMessages.current.delete(message_id)
          
          // Check for duplicate message content
          const messageContentKey = finalMessage.content.trim().substring(0, 100) // Use first 100 chars as key
          if (!recentMessages.has(messageContentKey)) {
            console.log('📝 MESSAGE: Processing new message content')
            
            setMessages(prev => prev.map(msg => 
              msg.id === message_id ? finalMessage : msg
            ))
            
            // Track this message content
            setRecentMessages(prev => {
              const newSet = new Set([...prev, messageContentKey])
              // Keep only recent messages (max 10)
              if (newSet.size > 10) {
                const recentArray = Array.from(newSet)
                return new Set(recentArray.slice(-10))
              }
              return newSet
            })
          } else {
            console.log('📝 MESSAGE: Skipping duplicate message content:', messageContentKey)
            // Still update the message but don't add to tracking
            setMessages(prev => prev.map(msg => 
              msg.id === message_id ? finalMessage : msg
            ))
          }

          // Extract screenshot data from completed assistant messages (with deduplication)
          if (finalMessage.role === 'assistant' && finalMessage.content) {
            const screenshotMatch = finalMessage.content.match(/!\[.*?\]\(data:image\/[^)]+\)/g)
            if (screenshotMatch) {
              const screenshotUrl = currentWebsiteUrl || 'unknown'
              
              // Check if we've already processed a screenshot for this URL recently
              if (!processedScreenshots.has(screenshotUrl)) {
                console.log('📸 SCREENSHOT: Found new screenshot data in completed message')
                const base64Match = screenshotMatch[0].match(/data:image\/[^)]+/)
                if (base64Match) {
                  const screenshotData = {
                    base64Data: base64Match[0],
                    url: screenshotUrl,
                    title: 'AI Generated Screenshot',
                    timestamp: Date.now()
                  }
                  console.log('📸 SCREENSHOT: Setting screenshot data for URL:', screenshotUrl)
                  setScreenshotData(screenshotData)
                  
                  // Mark this URL as processed
                  setProcessedScreenshots(prev => new Set([...prev, screenshotUrl]))
                  
                  // Clear screenshot data after a delay
                  setTimeout(() => {
                    setScreenshotData(null)
                  }, 1000)
                }
              } else {
                console.log('📸 SCREENSHOT: Skipping duplicate screenshot for URL:', screenshotUrl)
              }
            }
          }
        }
        break

      case 'tool_execution_update':
        // Update tool execution status
        const toolMessage = streamingMessages.current.get(message_id)
        if (toolMessage) {
          const updatedToolMessage = {
            ...toolMessage,
            content: progress_message || toolMessage.content,
            tool_status: status as any
          }
          streamingMessages.current.set(message_id, updatedToolMessage)
          
          setMessages(prev => prev.map(msg => 
            msg.id === message_id ? updatedToolMessage : msg
          ))

          // If tool completed successfully and it's a browser tool, check for split-screen
          if (status === 'completed' && toolMessage.tool_name?.includes('browser')) {
            console.log('🔧 FRONTEND FALLBACK: Browser tool completed:', toolMessage.tool_name)
            console.log('🔧 FRONTEND FALLBACK: Tool result:', streamingMessage.result)
            
            // Extract URL from result if available
            const result = streamingMessage.result
            if (result?.page_url || result?.url || result?.website_url) {
              const websiteUrl = result.page_url || result.url || result.website_url
              console.log('🎯 FRONTEND FALLBACK: Activating split-screen via fallback for:', websiteUrl)
              
              setCurrentWebsiteUrl(websiteUrl)
              setSplitScreenMode(true)
              
              // Add a system message
              const fallbackBrowserMessage: Message = {
                id: 'fallback-browser-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9),
                role: 'system',
                content: `🌐 Browser opened via fallback for ${websiteUrl} (${toolMessage.tool_name})`,
                timestamp: new Date().toISOString()
              }
              setMessages(prev => [...prev, fallbackBrowserMessage])
              
              console.log('✅ FRONTEND FALLBACK: Split-screen activated successfully')
            } else {
              console.log('⚠️ FRONTEND FALLBACK: Browser tool completed but no URL found')
              console.log('⚠️ FRONTEND FALLBACK: Available result fields:', Object.keys(result || {}))
            }
          }
        }
        break

      case 'workflow_status':
        // Show workflow status as system message
        const workflowMessage: Message = {
          id: 'workflow-' + Date.now(),
          role: 'system',
          content: `${streamingMessage.workflow_stage}: ${streamingMessage.status_message}`,
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev, workflowMessage])
        break

      case 'workflow_step':
        // Show detailed workflow step
        const { step_id, step_name, step_description, metadata } = streamingMessage
        const stepMessage: Message = {
          id: step_id,
          role: 'workflow',
          content: step_description,
          timestamp: new Date().toISOString(),
          workflow_step: step_name,
          workflow_status: status || 'in_progress',
          workflow_metadata: metadata,
          is_streaming: status === 'in_progress'
        }
        streamingMessages.current.set(step_id, stepMessage)
        setMessages(prev => [...prev, stepMessage])
        setStreamingStats(prev => ({ ...prev, workflowSteps: prev.workflowSteps + 1 }))
        break

      case 'workflow_step_update':
        // Update workflow step
        const stepUpdate = streamingMessages.current.get(streamingMessage.step_id)
        if (stepUpdate) {
          const updatedStep = {
            ...stepUpdate,
            workflow_status: streamingMessage.status,
            content: streamingMessage.result_description || stepUpdate.content,
            is_streaming: streamingMessage.status === 'in_progress',
            workflow_metadata: { ...stepUpdate.workflow_metadata, ...streamingMessage.metadata }
          }
          streamingMessages.current.set(streamingMessage.step_id, updatedStep)
          
          setMessages(prev => prev.map(msg => 
            msg.id === streamingMessage.step_id ? updatedStep : msg
          ))
        }
        break

      case 'browser_action':
        // Handle browser action to automatically trigger split-screen mode
        console.log('🔧 FRONTEND DEBUG: browser_action message received')
        console.log('🔧 FRONTEND DEBUG: Full message data:', streamingMessage)
        console.log('🔧 FRONTEND DEBUG: enable_split_screen:', streamingMessage.enable_split_screen)
        console.log('🔧 FRONTEND DEBUG: website_url:', streamingMessage.website_url)
        console.log('🔧 FRONTEND DEBUG: Current splitScreenMode:', splitScreenMode)
        console.log('🔧 FRONTEND DEBUG: Current currentWebsiteUrl:', currentWebsiteUrl)
        
        if (streamingMessage.enable_split_screen && streamingMessage.website_url) {
          console.log('🎯 FRONTEND: Activating split-screen mode!')
          console.log('🎯 FRONTEND: Setting website URL to:', streamingMessage.website_url)
          
          setCurrentWebsiteUrl(streamingMessage.website_url)
          setSplitScreenMode(true)
          
          console.log('✅ FRONTEND: Split-screen state updated')
          
          // Add a system message about the browser opening
          const browserMessage: Message = {
            id: 'browser-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9),
            role: 'system',
            content: `🌐 Opening browser for ${streamingMessage.website_url} ${streamingMessage.tool_name ? `(via ${streamingMessage.tool_name})` : ''}`,
            timestamp: new Date().toISOString()
          }
          setMessages(prev => [...prev, browserMessage])
          console.log('✅ FRONTEND: Added browser system message')
        } else {
          console.log('⚠️ FRONTEND: browser_action message missing required fields')
          console.log('⚠️ FRONTEND: enable_split_screen:', streamingMessage.enable_split_screen)
          console.log('⚠️ FRONTEND: website_url:', streamingMessage.website_url)
        }
        break

      case 'error':
        console.error('Streaming error:', streamingMessage.message)
        break
    }
  }

  const checkSessionStatus = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/user/session-status?session_id=${sessionId}`)
      const result = await response.json()
      
      if (result.success) {
        setIsSetupComplete(result.data.ready_to_use)
        
        if (!result.data.ready_to_use) {
          setMessages([{
            id: 'setup-' + Date.now(),
            role: 'system',
            content: `Welcome to MCP Router! 🚀\n\nTo get started, you'll need to:\n${!result.data.has_preferences ? '• Configure your LLM preferences\n' : ''}${!result.data.has_preferred_key ? '• Add API keys for your preferred provider\n' : ''}Click the settings icon (⚙️) in the top right to configure your account.`,
            timestamp: new Date().toISOString()
          }])
        }
      }
    } catch (error) {
      console.error('Error checking session status:', error)
      setMessages([{
        id: 'error-' + Date.now(),
        role: 'system',
        content: 'Unable to connect to backend. Please ensure the server is running on http://localhost:8000',
        timestamp: new Date().toISOString()
      }])
    }
  }

  const loadConversationHistory = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/user/conversation-history?session_id=${sessionId}&limit=50`)
      const result = await response.json()
      
      if (result.success && result.data.length > 0) {
        const historyMessages: Message[] = result.data.map((msg: any, index: number) => ({
          id: `history-${index}`,
          role: msg.message_type === 'user' ? 'user' : 'assistant',
          content: msg.content,
          timestamp: msg.created_at,
          tools_used: msg.tool_execution_data?.tools_used
        }))
        
        setMessages(prev => [...prev, ...historyMessages])
      }
    } catch (error) {
      console.error('Error loading conversation history:', error)
    }
  }

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    // Clear tracking when user sends new message
    setProcessedScreenshots(new Set())
    setRecentMessages(new Set())
    console.log('🔄 RESET: Cleared screenshot and message tracking for new user input')

    const userMessage: Message = {
      id: 'user-' + Date.now(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/v1/query/orchestrate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          user_id: sessionId,
          preferences: {
            session_id: sessionId,
            enable_streaming: true // Enable streaming for this session
          }
        }),
      })

      const result = await response.json()

      if (result.success) {
        // With streaming enabled, the response will come via WebSocket
        // So we don't add the assistant message here - it will be streamed
        
        // However, we still check for browser metadata for split-screen
        if (result.data.website_url && result.data.enable_split_screen) {
          setCurrentWebsiteUrl(result.data.website_url)
          setSplitScreenMode(true)
        }
      } else {
        const errorMessage: Message = {
          id: 'error-' + Date.now(),
          role: 'system',
          content: `Error: ${result.error || 'Failed to get response from AI'}`,
          timestamp: new Date().toISOString()
        }

        setMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: 'error-' + Date.now(),
        role: 'system',
        content: 'Failed to send message. Please check your connection.',
        timestamp: new Date().toISOString()
      }

      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearHistory = async () => {
    try {
      await fetch(`http://localhost:8000/api/v1/user/conversation-history?session_id=${sessionId}`, {
        method: 'DELETE'
      })
      setMessages([])
      streamingMessages.current.clear()
      
      // Clear tracking when history is cleared
      setProcessedScreenshots(new Set())
      setRecentMessages(new Set())
      console.log('🔄 RESET: Cleared all tracking on history clear')
    } catch (error) {
      console.error('Error clearing history:', error)
    }
  }

  const handleSettingsUpdated = () => {
    checkSessionStatus()
  }

  // Manual testing debug functions
  const testSplitScreenManually = () => {
    console.log('🧪 MANUAL TEST: Activating split-screen manually')
    const testUrl = 'https://google.com'  // Use a real site for testing
    setCurrentWebsiteUrl(testUrl)
    setSplitScreenMode(true)
    
    // Add a test message
    const testMessage: Message = {
      id: 'manual-test-' + Date.now(),
      role: 'system',
      content: `🧪 Manual split-screen test activated with ${testUrl}`,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, testMessage])
    console.log('🧪 MANUAL TEST: Split-screen should now be visible with', testUrl)
  }

  const simulateBrowserAction = () => {
    console.log('🧪 SIMULATE: Simulating browser_action message')
    handleStreamingMessage({
      type: 'browser_action',
      action_type: 'browser_opened_manual_test',
      website_url: 'https://google.com',
      enable_split_screen: true,
      tool_name: 'manual_test',
      timestamp: Date.now(),
      session_id: sessionId,
      user_id: sessionId
    })
  }

  const getWorkflowStepIcon = (stepName: string, status: string) => {
    const iconMap: Record<string, string> = {
      'llm_thinking': '🤔',
      'llm_processing': '🧠',
      'llm_to_tools': '🔧',
      'tools_execution': '⚡',
      'individual_tool': '🔧',
      'tools_to_llm': '↩️'
    }
    
    const statusIcon = status === 'completed' ? '✅' : status === 'error' ? '❌' : '⏳'
    const stepIcon = iconMap[stepName] || '🔄'
    
    return `${statusIcon} ${stepIcon}`
  }

  const truncateUrl = (text: string) => {
    // Find URLs in the text and truncate them
    const urlRegex = /https?:\/\/[^\s)]+/g
    return text.replace(urlRegex, (url) => {
      if (url.length > 50) {
        return url.substring(0, 30) + '...' + url.substring(url.length - 15)
      }
      return url
    })
  }

  const getMessageClassName = (message: Message) => {
    switch (message.role) {
      case 'user':
        return "max-w-[80%] p-3 rounded-lg bg-blue-600 text-white"
      case 'system':
      case 'tool':
      case 'workflow':
        // Simplified styling for intermediate messages - no colored backgrounds, smaller text
        return "max-w-[90%] px-2 py-1 text-xs text-gray-400"
      default:
        return "max-w-[80%] p-3 rounded-lg bg-slate-700 text-white"
    }
  }

  const chatPanel = (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <div className="flex items-center space-x-3">
          <h1 className="text-2xl font-bold">MCP Router</h1>
          {connectionStatus === 'connected' && (
            <div className="flex items-center space-x-1">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-xs text-green-400">Live</span>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {currentWebsiteUrl && (
            <button
              onClick={() => setSplitScreenMode(!splitScreenMode)}
              className={`p-2 rounded-lg transition-colors ${
                splitScreenMode 
                  ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                  : 'bg-slate-800 hover:bg-slate-700'
              }`}
              title={splitScreenMode ? "Exit split-screen mode" : "Enter split-screen mode"}
            >
              <Monitor className="w-5 h-5" />
            </button>
          )}
          {/* Debug buttons - only in development */}
          {process.env.NODE_ENV === 'development' && (
            <>
              <button
                onClick={testSplitScreenManually}
                className="p-2 rounded-lg bg-yellow-600 hover:bg-yellow-700 transition-colors text-white text-xs"
                title="🧪 Test Split Screen Manually"
              >
                🧪
              </button>
              <button
                onClick={simulateBrowserAction}
                className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 transition-colors text-white text-xs"
                title="🚀 Simulate Browser Action"
              >
                🚀
              </button>
            </>
          )}
          <button
            onClick={clearHistory}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
            title="Clear conversation history"
          >
            <History className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
            title="Settings - Configure LLM & API Keys"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-400 mt-8">
            <h2 className="text-xl mb-2">Welcome to MCP Router! 🚀</h2>
            <p>Your AI agent interface with real-time streaming.</p>
            <p className="mt-2 text-sm">Type a message below to see live responses.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div className={getMessageClassName(message)}>
                {/* Simplified content display */}
                <div className="whitespace-pre-wrap">
                  {message.role === 'workflow' && message.workflow_step ? 
                    `${message.workflow_step.replace('_', ' ').toUpperCase()}` :
                    message.role === 'tool' && message.tool_name ?
                    `${message.tool_name}` :
                    message.role === 'user' ? message.content :
                    truncateUrl(message.content)
                  }
                  {message.is_streaming && (
                    <span className="inline-block w-2 h-4 bg-current ml-1 animate-blink">|</span>
                  )}
                </div>
                
                {/* Only show timestamp for non-user messages */}
                {message.role !== 'user' && (
                  <div className="text-xs opacity-30 mt-1">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-700 text-white p-3 rounded-lg">
              <div className="flex items-center space-x-2">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing your request...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-700">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              isSetupComplete 
                ? "Type your message..." 
                : "Complete setup first (click settings)"
            }
            disabled={isLoading || !isSetupComplete}
            className="flex-1 p-3 bg-slate-800 text-white rounded-lg border border-slate-600 focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputValue.trim() || !isSetupComplete}
            className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
        <div className="text-xs text-gray-400 mt-2 flex items-center justify-between">
          <div>
            Session ID: {sessionId.slice(0, 8)}... | Press Enter to send
            {splitScreenMode && currentWebsiteUrl && (
              <span className="ml-2">| 🌐 Split-screen active</span>
            )}
          </div>
          <div className="flex items-center space-x-4">
            {connectionStatus === 'connected' && (
              <>
                <div className="flex items-center space-x-1">
                  <Zap className="w-3 h-3" />
                  <span>Messages: {streamingStats.messagesReceived}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Activity className="w-3 h-3" />
                  <span>Tools: {streamingStats.toolsExecuted}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Loader2 className="w-3 h-3" />
                  <span>Steps: {streamingStats.workflowSteps}</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )

  const websitePanel = currentWebsiteUrl ? (
    <div className="h-full flex flex-col">
      {/* Browser Shell - takes most of the space */}
      <div className="flex-1 min-h-0">
        <BrowserShell
          initialUrl={currentWebsiteUrl}
          userId={sessionId}
          onUrlChange={setCurrentWebsiteUrl}
          className="h-full"
          screenshotData={screenshotData || undefined}
        />
      </div>
      
      {/* MCP Tools Panel - fixed height at bottom */}
      <div className="flex-shrink-0">
        <MCPToolsPanel 
          messages={messages}
          className="border-l border-slate-600"
        />
      </div>
    </div>
  ) : null

  return (
    <div className="h-full">
      {splitScreenMode && currentWebsiteUrl ? (
        <SplitScreenLayout
          leftPanel={chatPanel}
          rightPanel={websitePanel}
          defaultSplit={splitPercentage}
          onSplitChange={setSplitPercentage}
          className="h-full"
        />
      ) : (
        chatPanel
      )}

      {/* Settings Panel */}
      <SettingsPanel
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        sessionId={sessionId}
        onSettingsUpdated={handleSettingsUpdated}
      />
      
      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
        .animate-blink {
          animation: blink 1s infinite;
        }
      `}</style>
    </div>
  )
}