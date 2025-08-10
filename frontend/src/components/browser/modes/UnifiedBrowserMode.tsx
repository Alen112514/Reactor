'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  Activity, 
  ExternalLink, 
  Loader2, 
  AlertCircle,
  Monitor,
  Bot,
  MousePointer,
  Eye
} from 'lucide-react'
import type { BrowserSession } from '../BrowserShell'

interface UnifiedBrowserModeProps {
  session: BrowserSession
  userId: string
  onSessionUpdate: (updates: Partial<BrowserSession>) => void
  onError?: (error: string) => void
  isFullscreen: boolean
  externalScreenshotData?: {
    base64Data: string
    url: string
    title?: string
    timestamp?: number
  }
}

interface BrowserFrame {
  frame_data: string
  page_url: string
  page_title: string
  timestamp: number
}

export default function UnifiedBrowserMode({
  session,
  userId,
  onSessionUpdate,
  onError,
  isFullscreen,
  externalScreenshotData
}: UnifiedBrowserModeProps) {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [currentFrame, setCurrentFrame] = useState<BrowserFrame | null>(null)
  const [isLLMActive, setIsLLMActive] = useState(false)
  const [llmActionDescription, setLlmActionDescription] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [connectionStats, setConnectionStats] = useState({
    frames_received: 0,
    last_frame_time: 0
  })

  const websocketRef = useRef<WebSocket | null>(null)
  const frameCountRef = useRef(0)

  // Handle external screenshot data (from MCP tools)
  useEffect(() => {
    if (externalScreenshotData) {
      console.log('📸 UnifiedBrowser: Received external screenshot data')
      
      // Display the screenshot as current frame
      setCurrentFrame({
        frame_data: externalScreenshotData.base64Data,
        page_url: externalScreenshotData.url,
        page_title: externalScreenshotData.title || 'AI Generated Screenshot',
        timestamp: externalScreenshotData.timestamp || Date.now()
      })
      
      // Update session info
      onSessionUpdate({
        url: externalScreenshotData.url,
        title: externalScreenshotData.title,
        isLoading: false,
        error: undefined
      })
    }
  }, [externalScreenshotData])

  // Connect to unified browser WebSocket
  useEffect(() => {
    if (session.url && session.url !== 'about:blank') {
      connectWebSocket()
    }

    return () => {
      disconnectWebSocket()
    }
  }, [session.url, userId])

  const connectWebSocket = async () => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setIsConnecting(true)
    setError(null)

    try {
      const wsUrl = `ws://localhost:8000/api/v1/unified-browser/ws/${userId}`
      console.log('🔗 Connecting to unified browser WebSocket:', wsUrl)

      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('✅ Unified browser WebSocket connected')
        setIsConnected(true)
        setIsConnecting(false)
        
        // Request browser session creation with current URL for session discovery
        ws.send(JSON.stringify({
          type: 'create_browser_session',
          current_url: session.url
        }))
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleWebSocketMessage(data)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onclose = () => {
        console.log('🔌 Unified browser WebSocket disconnected')
        setIsConnected(false)
        setIsConnecting(false)
        
        // Auto-reconnect after delay
        setTimeout(() => {
          if (session.url !== 'about:blank') {
            connectWebSocket()
          }
        }, 3000)
      }

      ws.onerror = (error) => {
        console.error('Unified browser WebSocket error:', error)
        setError('WebSocket connection failed')
        setIsConnected(false)
        setIsConnecting(false)
      }

      websocketRef.current = ws

    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      setError('Failed to establish connection')
      setIsConnecting(false)
    }
  }

  const disconnectWebSocket = () => {
    if (websocketRef.current) {
      websocketRef.current.close()
      websocketRef.current = null
    }
    setIsConnected(false)
    setIsConnecting(false)
  }

  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'connection_established':
        console.log('✅ Unified browser connection established:', data.connection_id)
        break

      case 'browser_session_created':
        console.log('🌐 Browser session created:', data.session_id)
        break

      case 'mcp_session_created':
        console.log('🚀 MCP session created and broadcast:', data.session_id)
        // MCP tool created a browser session, frontend should connect immediately
        setIsConnected(true)
        setError(null)
        break

      case 'browser_frame':
        // Received live browser frame
        setCurrentFrame({
          frame_data: data.frame_data,
          page_url: data.page_url,
          page_title: data.page_title,
          timestamp: data.timestamp
        })
        
        frameCountRef.current++
        setConnectionStats(prev => ({
          frames_received: prev.frames_received + 1,
          last_frame_time: data.timestamp
        }))

        // Update session with current page info
        onSessionUpdate({
          url: data.page_url,
          title: data.page_title,
          isLoading: false,
          error: undefined
        })
        break

      case 'llm_action_start':
        console.log('🤖 LLM action started:', data.task_description)
        setIsLLMActive(true)
        setLlmActionDescription(data.task_description)
        break

      case 'llm_action_complete':
        console.log('✅ LLM action completed')
        setIsLLMActive(false)
        setLlmActionDescription('')
        break

      case 'llm_action_error':
        console.log('❌ LLM action error:', data.error)
        setIsLLMActive(false)
        setLlmActionDescription('')
        setError(data.error)
        onError?.(data.error)
        break

      case 'error':
        console.error('WebSocket error:', data.message)
        setError(data.message)
        break
    }
  }

  const handleExternalRedirect = (e: React.MouseEvent) => {
    e.preventDefault()
    
    if (currentFrame?.page_url) {
      // Open current URL in external browser tab
      window.open(currentFrame.page_url, '_blank', 'noopener,noreferrer')
      
      // Show user feedback
      console.log(`🔗 Opening ${currentFrame.page_url} in external browser`)
      
      // Could add a toast notification here
      // toast.info(`Opening ${currentFrame.page_url} in external browser`)
    }
  }

  if (session.url === 'about:blank') {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-900 text-gray-400">
        <div className="text-center">
          <Monitor className="w-16 h-16 mx-auto mb-4 text-gray-600" />
          <h3 className="text-lg font-medium text-gray-300 mb-2">Unified Browser Mode</h3>
          <p className="text-sm">Navigate to a website to start live streaming</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-slate-900">
      {/* Status bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
        <div className="flex items-center space-x-4">
          {/* Connection status */}
          <div className="flex items-center space-x-2">
            {isConnecting ? (
              <Loader2 className="w-4 h-4 animate-spin text-yellow-500" />
            ) : isConnected ? (
              <Activity className="w-4 h-4 text-green-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-red-500" />
            )}
            <span className="text-sm text-gray-400">
              {isConnecting ? 'Connecting...' : isConnected ? 'Live Stream' : 'Disconnected'}
            </span>
          </div>

          {/* LLM activity indicator */}
          {isLLMActive && (
            <div className="flex items-center space-x-2 bg-blue-600/20 px-3 py-1 rounded-lg">
              <Bot className="w-4 h-4 text-blue-400 animate-pulse" />
              <span className="text-xs text-blue-300">
                LLM: {llmActionDescription}
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-4">
          {/* Frame stats */}
          {isConnected && (
            <div className="text-xs text-gray-500">
              Frames: {connectionStats.frames_received}
            </div>
          )}

          {/* Click instruction */}
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <MousePointer className="w-3 h-3" />
            <span>Click to open externally</span>
          </div>
        </div>
      </div>

      {/* Main browser display */}
      <div className="flex-1 relative min-h-0">
        {error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-red-400 max-w-md">
              <AlertCircle className="w-16 h-16 mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Connection Error</h3>
              <p className="text-sm text-gray-300">{error}</p>
            </div>
          </div>
        ) : currentFrame ? (
          <div className="absolute inset-0 w-full h-full">
            {/* Browser frame display */}
            <img
              src={currentFrame.frame_data}
              alt={`Browser content: ${currentFrame.page_title}`}
              className="w-full h-full object-contain bg-white"
              style={{ minHeight: '400px' }}
            />

            {/* Click overlay for external redirect */}
            <div
              className="absolute inset-0 cursor-pointer z-10"
              onClick={handleExternalRedirect}
              title={`Click to open ${currentFrame.page_url} in external browser`}
            />

            {/* LLM activity overlay */}
            {isLLMActive && (
              <div className="absolute top-4 left-4 bg-blue-600/90 text-white px-4 py-2 rounded-lg shadow-lg">
                <div className="flex items-center space-x-2">
                  <Bot className="w-5 h-5 animate-pulse" />
                  <div>
                    <div className="text-sm font-medium">LLM Active</div>
                    <div className="text-xs opacity-90">{llmActionDescription}</div>
                  </div>
                </div>
              </div>
            )}

            {/* External link indicator */}
            <div className="absolute top-4 right-4 bg-slate-800/90 text-gray-300 px-3 py-2 rounded-lg shadow-lg">
              <div className="flex items-center space-x-2">
                <ExternalLink className="w-4 h-4" />
                <span className="text-xs">Click anywhere to open externally</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <Eye className="w-16 h-16 mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Waiting for Content</h3>
              <p className="text-sm">
                {isConnected ? 'Waiting for browser frames...' : 'Connecting to browser...'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}