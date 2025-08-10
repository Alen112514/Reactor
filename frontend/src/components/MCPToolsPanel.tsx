'use client'

import { useState, useEffect } from 'react'
import { 
  Wrench, 
  Activity, 
  CheckCircle, 
  XCircle, 
  Clock, 
  ChevronDown, 
  ChevronRight,
  Bot,
  Zap,
  Info
} from 'lucide-react'

interface MCPTool {
  id: string
  name: string
  status: 'starting' | 'running' | 'completed' | 'error'
  description?: string
  startTime: number
  endTime?: number
  progress?: string
  error?: string
  metadata?: any
}

interface MCPToolsPanelProps {
  className?: string
  messages?: any[] // Messages from StreamingChatInterface to extract tool info
}

export default function MCPToolsPanel({ className = '', messages = [] }: MCPToolsPanelProps) {
  const [activeMCPTools, setActiveMCPTools] = useState<MCPTool[]>([])
  const [isExpanded, setIsExpanded] = useState(true)
  const [recentTools, setRecentTools] = useState<MCPTool[]>([])

  // Extract MCP tool information from messages
  useEffect(() => {
    const toolMessages = messages.filter(msg => 
      msg.role === 'tool' || 
      msg.role === 'workflow' || 
      (msg.content && (msg.content.includes('MCP') || msg.content.includes('tool')))
    )

    const tools: MCPTool[] = []
    const completed: MCPTool[] = []

    toolMessages.forEach(msg => {
      if (msg.role === 'tool' && msg.tool_name) {
        const tool: MCPTool = {
          id: msg.id,
          name: msg.tool_name,
          status: msg.tool_status || 'running',
          description: msg.content,
          startTime: new Date(msg.timestamp).getTime(),
          progress: msg.content
        }

        if (msg.tool_status === 'completed' || msg.tool_status === 'error') {
          tool.endTime = Date.now()
          completed.push(tool)
        } else {
          tools.push(tool)
        }
      } else if (msg.role === 'workflow' && msg.workflow_step) {
        const tool: MCPTool = {
          id: msg.id,
          name: msg.workflow_step.replace('_', ' ').toUpperCase(),
          status: msg.workflow_status === 'completed' ? 'completed' : 
                  msg.workflow_status === 'error' ? 'error' : 'running',
          description: msg.content,
          startTime: new Date(msg.timestamp).getTime(),
          progress: msg.content,
          metadata: msg.workflow_metadata
        }

        if (tool.status === 'completed' || tool.status === 'error') {
          tool.endTime = Date.now()
          completed.push(tool)
        } else {
          tools.push(tool)
        }
      }
    })

    setActiveMCPTools(tools)
    setRecentTools(completed.slice(-5)) // Keep last 5 completed tools
  }, [messages])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'starting':
        return <Clock className="w-4 h-4 text-blue-400" />
      case 'running':
        return <Activity className="w-4 h-4 text-orange-400 animate-spin" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'error':
        return <XCircle className="w-4 h-4 text-red-400" />
      default:
        return <Wrench className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'starting':
        return 'text-blue-400 bg-blue-400/10'
      case 'running':
        return 'text-orange-400 bg-orange-400/10'
      case 'completed':
        return 'text-green-400 bg-green-400/10'
      case 'error':
        return 'text-red-400 bg-red-400/10'
      default:
        return 'text-gray-400 bg-gray-400/10'
    }
  }

  const formatDuration = (startTime: number, endTime?: number) => {
    const duration = (endTime || Date.now()) - startTime
    const seconds = Math.floor(duration / 1000)
    const minutes = Math.floor(seconds / 60)
    
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`
    }
    return `${seconds}s`
  }

  const hasAnyTools = activeMCPTools.length > 0 || recentTools.length > 0

  return (
    <div className={`bg-slate-800 border-t border-slate-700 ${className}`}>
      {/* Header */}
      <div 
        className="flex items-center justify-between px-4 py-2 cursor-pointer hover:bg-slate-700 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <Bot className="w-4 h-4 text-blue-400" />
          <span className="text-sm font-medium text-gray-200">MCP Tools</span>
          {activeMCPTools.length > 0 && (
            <span className="bg-orange-500 text-white text-xs px-2 py-1 rounded-full">
              {activeMCPTools.length} active
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {activeMCPTools.some(tool => tool.status === 'running') && (
            <Zap className="w-3 h-3 text-orange-400 animate-pulse" />
          )}
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-3 max-h-48 overflow-y-auto">
          {!hasAnyTools ? (
            <div className="text-center py-6 text-gray-500">
              <Info className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No active MCP tools</p>
              <p className="text-xs opacity-75">Tools will appear here when running</p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Active Tools */}
              {activeMCPTools.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
                    Active Tools
                  </h4>
                  <div className="space-y-2">
                    {activeMCPTools.map((tool) => (
                      <div
                        key={tool.id}
                        className={`flex items-start space-x-3 p-2 rounded-lg ${getStatusColor(tool.status)}`}
                      >
                        <div className="flex-shrink-0 mt-0.5">
                          {getStatusIcon(tool.status)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-medium text-gray-200 truncate">
                              {tool.name}
                            </p>
                            <span className="text-xs text-gray-400 flex-shrink-0">
                              {formatDuration(tool.startTime)}
                            </span>
                          </div>
                          {tool.progress && (
                            <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                              {tool.progress}
                            </p>
                          )}
                          {tool.metadata?.tool_names && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {tool.metadata.tool_names.map((toolName: string, idx: number) => (
                                <span 
                                  key={idx}
                                  className="text-xs bg-slate-700 text-gray-300 px-1.5 py-0.5 rounded"
                                >
                                  {toolName}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recent Completed Tools */}
              {recentTools.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
                    Recently Completed
                  </h4>
                  <div className="space-y-1">
                    {recentTools.slice(-3).map((tool) => (
                      <div
                        key={tool.id}
                        className="flex items-center justify-between p-2 bg-slate-700/50 rounded"
                      >
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(tool.status)}
                          <span className="text-sm text-gray-300">{tool.name}</span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {formatDuration(tool.startTime, tool.endTime)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}