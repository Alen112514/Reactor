'use client'

import { useState, useEffect } from 'react'
import { 
  Monitor, 
  Globe,
  AlertCircle, 
  Loader2, 
  Maximize2
} from 'lucide-react'
import UnifiedBrowserMode from './modes/UnifiedBrowserMode'
import type { BrowserSession } from './BrowserShell'

interface BrowserContentProps {
  session?: BrowserSession
  userId: string
  onSessionUpdate: (updates: Partial<BrowserSession>) => void
  onError?: (error: string) => void
  externalScreenshotData?: {
    base64Data: string
    url: string
    title?: string
    timestamp?: number
  }
}

export default function BrowserContent({
  session,
  userId,
  onSessionUpdate,
  onError,
  externalScreenshotData
}: BrowserContentProps) {
  // Locked to unified mode only
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Always use unified mode
  useEffect(() => {
    if (!session) return

    // Force unified mode for all sessions
    if (session.mode !== 'unified') {
      onSessionUpdate({ mode: 'unified' })
    }
  }, [session?.url, session?.mode, onSessionUpdate])

  const handleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  const renderModeSelector = () => (
    <div className="flex items-center space-x-1 bg-slate-800 rounded-lg p-1">
      <div className="flex items-center space-x-1 px-3 py-1.5 rounded-md text-xs bg-blue-600 text-white">
        <Monitor className="w-3 h-3" />
        <span>Unified Browser</span>
      </div>
    </div>
  )

  const renderContentControls = () => (
    <div className="flex items-center space-x-2">
      {renderModeSelector()}
      
      <div className="h-4 w-px bg-slate-600"></div>
      
      <button
        onClick={handleFullscreen}
        className="p-1.5 rounded-md hover:bg-slate-700 transition-colors text-gray-400 hover:text-white"
        title="Toggle fullscreen"
      >
        <Maximize2 className="w-4 h-4" />
      </button>
    </div>
  )

  if (!session) {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-900 text-gray-400">
        <div className="text-center">
          <Globe className="w-16 h-16 mx-auto mb-4 text-gray-600" />
          <h3 className="text-lg font-medium text-gray-300 mb-2">No Active Session</h3>
          <p className="text-sm">Open a URL to start browsing</p>
        </div>
      </div>
    )
  }

  if (session.url === 'about:blank') {
    return (
      <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="text-center">
          <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center">
            <Globe className="w-12 h-12 text-blue-400" />
          </div>
          <h3 className="text-xl font-medium text-gray-300 mb-2">New Tab</h3>
          <p className="text-sm text-gray-500">Enter a URL in the address bar to get started</p>
        </div>
      </div>
    )
  }

  const containerClasses = isFullscreen 
    ? 'fixed inset-0 z-50 bg-black'
    : 'h-full flex flex-col relative'

  return (
    <div className={containerClasses}>
      {/* Content controls bar */}
      {!isFullscreen && (
        <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
          <div className="flex items-center space-x-3">
            <span className="text-xs text-gray-400">Display Mode:</span>
            {renderContentControls()}
          </div>
          
          <div className="flex items-center space-x-2 text-xs text-gray-400">
            {session.isLoading && (
              <div className="flex items-center space-x-1">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Loading...</span>
              </div>
            )}
            {session.error && (
              <div className="flex items-center space-x-1 text-red-400">
                <AlertCircle className="w-3 h-3" />
                <span>Error</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Content area */}
      <div className="flex-1 relative bg-white min-h-0">
        {/* Always render unified browser mode */}
        <UnifiedBrowserMode
          session={session}
          userId={userId}
          onSessionUpdate={onSessionUpdate}
          onError={onError}
          isFullscreen={isFullscreen}
          externalScreenshotData={externalScreenshotData}
        />

        {/* Fullscreen exit button */}
        {isFullscreen && (
          <button
            onClick={() => setIsFullscreen(false)}
            className="absolute top-4 right-4 z-10 p-2 bg-black/50 hover:bg-black/70 rounded-lg text-white transition-colors"
            title="Exit fullscreen"
          >
            <Maximize2 className="w-5 h-5" />
          </button>
        )}
      </div>
    </div>
  )
}