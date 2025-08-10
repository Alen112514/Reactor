'use client'

import React, { useState, useEffect } from 'react'
// No icon imports needed for text-only interface

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
  const [isExpanded, setIsExpanded] = useState(false)
  const [recentTools, setRecentTools] = useState<MCPTool[]>([])
  const [toolExecutionDetailsExpanded, setToolExecutionDetailsExpanded] = useState(false)
  const [detailedExecutions, setDetailedExecutions] = useState<any[]>([])

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
    
    // Extract detailed execution information for workflow and tool messages
    const detailedExecs = messages.filter(msg => 
      (msg.role === 'workflow' && msg.workflow_step) ||
      (msg.role === 'tool' && msg.tool_name) ||
      (msg.role === 'system' && msg.content?.includes('browser'))
    ).map(msg => ({
      ...msg,
      displayName: msg.workflow_step ? 
        msg.workflow_step.replace('_', ' ').toUpperCase() : 
        (msg.tool_name || 'System Action'),
      stepType: msg.workflow_step || 'tool_execution',
      fullUrl: msg.content?.match(/https?:\/\/[^\s)]+/g)?.[0] || null
    })).slice(-10) // Keep last 10 detailed executions
    
    setDetailedExecutions(detailedExecs)
  }, [messages])

  const getStatusText = (status: string) => {
    switch (status) {
      case 'starting':
        return '[Starting]'
      case 'running':
        return '[Running]'
      case 'completed':
        return '[Done]'
      case 'error':
        return '[Error]'
      default:
        return '[Pending]'
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

  const truncateUrl = (url: string, maxLength: number = 50) => {
    if (!url || url.length <= maxLength) return url
    return url.substring(0, maxLength - 3) + '...'
  }

  return (
    <div className={`bg-slate-800 border-t border-slate-700 ${className}`}>
      {/* Header */}
      <div 
        className="flex items-center justify-between px-4 py-2 cursor-pointer hover:bg-slate-700 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-gray-200">
            MCP Tools {isExpanded ? '[-]' : '[+]'}
          </span>
          {activeMCPTools.length > 0 && (
            <span className="text-xs text-gray-400">
              ({activeMCPTools.length} active)
            </span>
          )}
        </div>
        <div className="text-xs text-gray-400">
          {activeMCPTools.some(tool => tool.status === 'running') && 'Running...'}
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-3 max-h-48 overflow-y-auto">
          {!hasAnyTools ? (
            <div className="text-center py-6 text-gray-500">
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
                        className={`p-2 rounded-lg ${getStatusColor(tool.status)}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-gray-300">{getStatusText(tool.status)}</span>
                            <span className="text-sm font-medium text-gray-200 truncate">
                              {tool.name}
                            </span>
                          </div>
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
                          <span className="text-xs text-gray-400">{getStatusText(tool.status)}</span>
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

      {/* Tool Execution Details Section */}
      <div className="border-t border-slate-600">
        {/* Execution Details Header */}
        <div 
          className="flex items-center justify-between px-4 py-2 cursor-pointer hover:bg-slate-700 transition-colors"
          onClick={() => setToolExecutionDetailsExpanded(!toolExecutionDetailsExpanded)}
        >
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-200">
              Tool Execution Details {toolExecutionDetailsExpanded ? '[-]' : '[+]'}
            </span>
            {detailedExecutions.length > 0 && (
              <span className="text-xs text-gray-400">
                ({detailedExecutions.length})
              </span>
            )}
          </div>
        </div>

        {/* Execution Details Content */}
        {toolExecutionDetailsExpanded && (
          <div className="px-4 pb-3 max-h-64 overflow-y-auto">
            {detailedExecutions.length === 0 ? (
              <div className="text-center py-4 text-gray-500">
                <p className="text-xs">No execution details available</p>
                <p className="text-xs opacity-75">Details will appear when tools run</p>
              </div>
            ) : (
              <div className="space-y-2">
                {detailedExecutions.map((exec, index) => (
                  <div
                    key={exec.id || index}
                    className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/50"
                  >
                    {/* Execution Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-gray-200">
                          {exec.displayName}
                        </span>
                        {exec.workflow_status && (
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            exec.workflow_status === 'completed' ? 'bg-green-600 text-white' :
                            exec.workflow_status === 'error' ? 'bg-red-600 text-white' :
                            'bg-orange-600 text-white'
                          }`}>
                            {exec.workflow_status}
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-gray-400">
                        {new Date(exec.timestamp).toLocaleTimeString()}
                      </span>
                    </div>

                    {/* Execution Content */}
                    <div className="text-xs text-gray-300 space-y-2">
                      {/* Show full URL if available */}
                      {exec.fullUrl && (
                        <div className="bg-slate-800/50 rounded p-2">
                          <div className="font-medium text-blue-400 mb-1">URL:</div>
                          <div className="font-mono text-xs break-all">{exec.fullUrl}</div>
                        </div>
                      )}

                      {/* Show tool metadata */}
                      {exec.workflow_metadata && (
                        <div className="space-y-1">
                          {exec.workflow_metadata.tool_names && (
                            <div>
                              <span className="font-medium text-yellow-400">Tools:</span>{' '}
                              {exec.workflow_metadata.tool_names.join(', ')}
                            </div>
                          )}
                          {exec.workflow_metadata.parameters_count && (
                            <div>
                              <span className="font-medium text-blue-400">Parameters:</span>{' '}
                              {exec.workflow_metadata.parameters_count} items
                            </div>
                          )}
                        </div>
                      )}

                      {/* Show execution content (truncated) */}
                      {exec.content && (
                        <div className="text-gray-400">
                          {exec.content.length > 150 ? 
                            `${exec.content.substring(0, 150)}...` : 
                            exec.content
                          }
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}