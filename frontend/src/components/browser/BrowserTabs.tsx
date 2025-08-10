'use client'

import { useState } from 'react'
import { X, Plus, Loader2 } from 'lucide-react'
import type { BrowserSession } from './BrowserShell'

interface BrowserTabsProps {
  sessions: BrowserSession[]
  activeSessionId: string
  onSessionChange: (sessionId: string) => void
  onNewTab: () => void
  onCloseTab: (sessionId: string) => void
}

export default function BrowserTabs({
  sessions,
  activeSessionId,
  onSessionChange,
  onNewTab,
  onCloseTab
}: BrowserTabsProps) {
  const [draggedTabId, setDraggedTabId] = useState<string | null>(null)

  const getTabTitle = (session: BrowserSession) => {
    if (session.url === 'about:blank') return 'New Tab'
    if (session.isLoading) return 'Loading...'
    if (session.title && session.title !== 'Loading...') return session.title
    
    // Extract domain from URL as fallback
    try {
      const url = new URL(session.url)
      return url.hostname
    } catch {
      return session.url
    }
  }

  const getFaviconUrl = (session: BrowserSession) => {
    if (session.url === 'about:blank') return null
    
    try {
      const url = new URL(session.url)
      return `https://www.google.com/s2/favicons?domain=${url.hostname}&sz=16`
    } catch {
      return null
    }
  }

  const handleTabClick = (sessionId: string) => {
    onSessionChange(sessionId)
  }

  const handleTabClose = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    onCloseTab(sessionId)
  }

  const handleDragStart = (e: React.DragEvent, sessionId: string) => {
    setDraggedTabId(sessionId)
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleDragEnd = () => {
    setDraggedTabId(null)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }

  const handleDrop = (e: React.DragEvent, targetSessionId: string) => {
    e.preventDefault()
    // TODO: Implement tab reordering
    setDraggedTabId(null)
  }

  return (
    <div className="browser-tabs bg-slate-800 border-b border-slate-600">
      <div className="flex items-center overflow-x-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-transparent">
        {/* Tab List */}
        <div className="flex items-center flex-nowrap min-w-0">
          {sessions.map((session) => {
            const isActive = session.id === activeSessionId
            const faviconUrl = getFaviconUrl(session)
            const title = getTabTitle(session)
            
            return (
              <div
                key={session.id}
                draggable
                onDragStart={(e) => handleDragStart(e, session.id)}
                onDragEnd={handleDragEnd}
                onDragOver={handleDragOver}
                onDrop={(e) => handleDrop(e, session.id)}
                onClick={() => handleTabClick(session.id)}
                className={`
                  flex items-center min-w-0 max-w-64 px-4 py-2 cursor-pointer group
                  border-r border-slate-600 transition-all duration-200
                  ${isActive 
                    ? 'bg-slate-700 border-b-2 border-b-blue-500' 
                    : 'hover:bg-slate-700/50'
                  }
                  ${draggedTabId === session.id ? 'opacity-50' : ''}
                `}
              >
                {/* Favicon or loading indicator */}
                <div className="flex-shrink-0 w-4 h-4 mr-2">
                  {session.isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                  ) : faviconUrl ? (
                    <img
                      src={faviconUrl}
                      alt=""
                      className="w-4 h-4"
                      onError={(e) => {
                        // Hide favicon if it fails to load
                        e.currentTarget.style.display = 'none'
                      }}
                    />
                  ) : (
                    <div className="w-4 h-4 bg-slate-600 rounded-sm flex items-center justify-center">
                      <div className="w-2 h-2 bg-slate-400 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Tab title */}
                <span className="flex-1 min-w-0 text-sm text-gray-300 truncate">
                  {title}
                </span>

                {/* Close button */}
                <button
                  onClick={(e) => handleTabClose(e, session.id)}
                  className={`
                    flex-shrink-0 ml-2 p-0.5 rounded-full transition-colors
                    ${isActive 
                      ? 'hover:bg-slate-600 opacity-60 hover:opacity-100' 
                      : 'opacity-0 group-hover:opacity-60 hover:opacity-100 hover:bg-slate-600'
                    }
                  `}
                  title="Close tab"
                >
                  <X className="w-3 h-3" />
                </button>

                {/* Active tab indicator */}
                {isActive && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500"></div>
                )}
              </div>
            )
          })}
        </div>

        {/* New tab button */}
        <button
          onClick={onNewTab}
          className="flex-shrink-0 p-2 mx-2 rounded-full hover:bg-slate-700 transition-colors"
          title="New tab"
        >
          <Plus className="w-4 h-4 text-gray-400" />
        </button>

        {/* Tab actions */}
        <div className="flex-shrink-0 ml-auto pr-2">
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <span>{sessions.length} tab{sessions.length !== 1 ? 's' : ''}</span>
          </div>
        </div>
      </div>

      {/* Tab overflow shadow indicator */}
      {sessions.length > 6 && (
        <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-slate-800 to-transparent pointer-events-none"></div>
      )}
    </div>
  )
}