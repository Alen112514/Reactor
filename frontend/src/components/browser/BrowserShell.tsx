'use client'

import { useState, useEffect } from 'react'
import BrowserTopBar from './BrowserTopBar'
import BrowserTabs from './BrowserTabs'
import BrowserContent from './BrowserContent'
import BrowserStatus from './BrowserStatus'

export interface BrowserSession {
  id: string
  url: string
  title: string
  isActive: boolean
  isLoading: boolean
  isSecure: boolean
  mode: 'unified' | 'screenshot' | 'iframe'
  lastUpdated: number
  error?: string
  screenshots?: Array<{
    id: string
    url: string
    base64Data?: string
    timestamp: number
    title: string
  }>
}

interface BrowserShellProps {
  initialUrl?: string
  userId: string
  onUrlChange?: (url: string) => void
  onError?: (error: string) => void
  className?: string
  screenshotData?: {
    base64Data: string
    url: string
    title?: string
    timestamp?: number
  }
}

export default function BrowserShell({ 
  initialUrl = '', 
  userId, 
  onUrlChange, 
  onError,
  className = '',
  screenshotData
}: BrowserShellProps) {
  const [sessions, setSessions] = useState<BrowserSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string>('')
  const [isMinimized, setIsMinimized] = useState(false)
  const [isMaximized, setIsMaximized] = useState(false)

  // Initialize with first session if URL provided
  useEffect(() => {
    if (initialUrl && sessions.length === 0) {
      const firstSession: BrowserSession = {
        id: 'session-' + Date.now(),
        url: initialUrl,
        title: 'Loading...',
        isActive: true,
        isLoading: true,
        isSecure: initialUrl.startsWith('https://'),
        mode: 'unified',
        lastUpdated: Date.now()
      }
      setSessions([firstSession])
      setActiveSessionId(firstSession.id)
    }
  }, [initialUrl])

  // Handle incoming screenshot data and auto-switch to screenshot mode
  useEffect(() => {
    if (screenshotData && activeSession) {
      const newScreenshot = {
        id: 'screenshot-' + Date.now(),
        url: screenshotData.url,
        base64Data: screenshotData.base64Data,
        timestamp: screenshotData.timestamp || Date.now(),
        title: screenshotData.title || activeSession.title || 'Screenshot'
      }

      console.log('📸 BrowserShell: Received screenshot data')
      console.log('📸 BrowserShell: Screenshot URL:', screenshotData.url)
      console.log('📸 BrowserShell: Current session mode:', activeSession.mode)
      
      setSessions(prev => prev.map(session => 
        session.id === activeSessionId 
          ? { 
              ...session, 
              // Keep unified mode active - it handles screenshot data internally
              mode: 'unified', // Always use unified mode
              isLoading: false,
              error: undefined, // Clear any previous errors
              screenshots: [newScreenshot, ...(session.screenshots || [])],
              lastUpdated: Date.now()
            }
          : session
      ))

      console.log('📸 BrowserShell: Session updated with screenshot data')
    }
  }, [screenshotData, activeSessionId])

  const activeSession = sessions.find(s => s.id === activeSessionId)

  const handleUrlChange = (url: string) => {
    if (!activeSession) return
    
    setSessions(prev => prev.map(session => 
      session.id === activeSessionId 
        ? { 
            ...session, 
            url, 
            isLoading: true, 
            isSecure: url.startsWith('https://'),
            lastUpdated: Date.now()
          }
        : session
    ))
    onUrlChange?.(url)
  }

  const handleSessionUpdate = (updates: Partial<BrowserSession>) => {
    if (!activeSession) return
    
    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? { ...session, ...updates, lastUpdated: Date.now() }
        : session
    ))
  }

  const handleNewTab = () => {
    const newSession: BrowserSession = {
      id: 'session-' + Date.now(),
      url: 'about:blank',
      title: 'New Tab',
      isActive: false,
      isLoading: false,
      isSecure: false,
      mode: 'unified',
      lastUpdated: Date.now()
    }
    setSessions(prev => [...prev, newSession])
    setActiveSessionId(newSession.id)
  }

  const handleCloseTab = (sessionId: string) => {
    setSessions(prev => {
      const filtered = prev.filter(s => s.id !== sessionId)
      if (filtered.length === 0) {
        // If no tabs left, create a new blank one
        const blankSession: BrowserSession = {
          id: 'session-' + Date.now(),
          url: 'about:blank',
          title: 'New Tab',
          isActive: false,
          isLoading: false,
          isSecure: false,
          mode: 'unified', // Always use unified mode
          lastUpdated: Date.now()
        }
        setActiveSessionId(blankSession.id)
        return [blankSession]
      }
      
      // If we closed the active tab, switch to first available
      if (sessionId === activeSessionId) {
        setActiveSessionId(filtered[0].id)
      }
      return filtered
    })
  }

  const handleClose = () => {
    setIsMinimized(true)
  }

  const handleMinimize = () => {
    setIsMinimized(!isMinimized)
  }

  const handleMaximize = () => {
    setIsMaximized(!isMaximized)
  }

  if (isMinimized) {
    return (
      <div className={`browser-shell-minimized ${className}`}>
        <button 
          onClick={() => setIsMinimized(false)}
          className="flex items-center space-x-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
        >
          <div className="w-4 h-4 bg-gradient-to-br from-red-500 to-orange-500 rounded-full"></div>
          <span className="text-sm text-gray-300">Browser</span>
          {activeSession && (
            <span className="text-xs text-gray-500">({activeSession.title})</span>
          )}
        </button>
      </div>
    )
  }

  return (
    <div 
      className={`browser-shell ${isMaximized ? 'browser-shell-maximized' : ''} ${className} flex flex-col h-full`}
      style={{
        background: 'rgba(15, 23, 42, 0.95)',
        backdropFilter: 'blur(20px)',
        borderRadius: isMaximized ? '0px' : '12px',
        border: '1px solid rgba(148, 163, 184, 0.2)',
        boxShadow: isMaximized 
          ? 'none' 
          : '0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05)',
        overflow: 'hidden',
        transition: 'all 0.3s ease'
      }}
    >
      {/* Browser Top Bar */}
      <BrowserTopBar
        session={activeSession}
        onUrlChange={handleUrlChange}
        onClose={handleClose}
        onMinimize={handleMinimize}
        onMaximize={handleMaximize}
        onRefresh={() => handleSessionUpdate({ isLoading: true })}
        isMaximized={isMaximized}
      />

      {/* Browser Tabs */}
      <BrowserTabs
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSessionChange={setActiveSessionId}
        onNewTab={handleNewTab}
        onCloseTab={handleCloseTab}
      />

      {/* Browser Content - takes remaining space */}
      <div className="flex-1 min-h-0">
        <BrowserContent
          session={activeSession}
          userId={userId}
          onSessionUpdate={handleSessionUpdate}
          onError={onError}
          externalScreenshotData={screenshotData}
        />
      </div>

      {/* Browser Status Bar */}
      <BrowserStatus
        session={activeSession}
        totalSessions={sessions.length}
      />
    </div>
  )
}