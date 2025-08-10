'use client'

import { useState } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  RotateCcw,
  Shield,
  Lock,
  AlertTriangle,
  Home,
  Star,
  MoreHorizontal,
  Minimize2,
  Maximize2,
  X,
  ExternalLink
} from 'lucide-react'
import type { BrowserSession } from './BrowserShell'

interface BrowserTopBarProps {
  session?: BrowserSession
  onUrlChange: (url: string) => void
  onClose: () => void
  onMinimize: () => void
  onMaximize: () => void
  onRefresh: () => void
  isMaximized: boolean
}

export default function BrowserTopBar({
  session,
  onUrlChange,
  onClose,
  onMinimize,
  onMaximize,
  onRefresh,
  isMaximized
}: BrowserTopBarProps) {
  const [urlInput, setUrlInput] = useState(session?.url || '')
  const [isUrlBarFocused, setIsUrlBarFocused] = useState(false)

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!urlInput.trim()) return
    
    const finalUrl = urlInput.startsWith('http') ? urlInput : `https://${urlInput}`
    onUrlChange(finalUrl)
  }

  const handleExternalOpen = () => {
    if (session?.url) {
      window.open(session.url, '_blank', 'noopener,noreferrer')
    }
  }

  const getSecurityIcon = () => {
    if (!session) return <Shield className="w-4 h-4 text-gray-500" />
    
    if (session.isSecure) {
      return <Lock className="w-4 h-4 text-green-500" />
    } else if (session.url.startsWith('http://')) {
      return <AlertTriangle className="w-4 h-4 text-yellow-500" />
    }
    return <Shield className="w-4 h-4 text-gray-500" />
  }

  const getSecurityText = () => {
    if (!session) return 'No connection'
    
    if (session.isSecure) {
      return 'Secure connection'
    } else if (session.url.startsWith('http://')) {
      return 'Not secure'
    }
    return 'Unknown'
  }

  return (
    <div className="browser-top-bar bg-gradient-to-r from-slate-800 to-slate-700 border-b border-slate-600">
      {/* macOS-style window controls */}
      <div className="flex items-center justify-between px-4 py-3">
        {/* Left side - Traffic lights and navigation */}
        <div className="flex items-center space-x-3">
          {/* Traffic light buttons */}
          <div className="flex items-center space-x-2">
            <button
              onClick={onClose}
              className="w-3 h-3 bg-red-500 hover:bg-red-600 rounded-full transition-colors"
              title="Close"
            />
            <button
              onClick={onMinimize}
              className="w-3 h-3 bg-yellow-500 hover:bg-yellow-600 rounded-full transition-colors"
              title="Minimize"
            />
            <button
              onClick={onMaximize}
              className="w-3 h-3 bg-green-500 hover:bg-green-600 rounded-full transition-colors"
              title={isMaximized ? "Restore" : "Maximize"}
            />
          </div>

          {/* Navigation controls */}
          <div className="flex items-center space-x-1 ml-4">
            <button
              className="p-1.5 rounded-md hover:bg-slate-600 transition-colors disabled:opacity-50"
              title="Go back"
              disabled
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <button
              className="p-1.5 rounded-md hover:bg-slate-600 transition-colors disabled:opacity-50"
              title="Go forward"
              disabled
            >
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={onRefresh}
              className="p-1.5 rounded-md hover:bg-slate-600 transition-colors"
              title="Refresh"
            >
              <RotateCcw className={`w-4 h-4 ${session?.isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              className="p-1.5 rounded-md hover:bg-slate-600 transition-colors"
              title="Home"
            >
              <Home className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Center - URL bar */}
        <div className="flex-1 max-w-2xl mx-8">
          <form onSubmit={handleUrlSubmit} className="relative">
            <div 
              className={`flex items-center bg-slate-900 rounded-lg border transition-colors ${
                isUrlBarFocused 
                  ? 'border-blue-500 bg-slate-800' 
                  : 'border-slate-600 hover:border-slate-500'
              }`}
            >
              {/* Security indicator */}
              <div className="flex items-center px-3 py-2 border-r border-slate-600">
                <div className="flex items-center space-x-1" title={getSecurityText()}>
                  {getSecurityIcon()}
                  {session?.isSecure && (
                    <span className="text-xs text-green-400 font-medium">HTTPS</span>
                  )}
                </div>
              </div>

              {/* URL input */}
              <input
                type="text"
                value={isUrlBarFocused ? urlInput : session?.url || ''}
                onChange={(e) => setUrlInput(e.target.value)}
                onFocus={() => {
                  setIsUrlBarFocused(true)
                  setUrlInput(session?.url || '')
                }}
                onBlur={() => setIsUrlBarFocused(false)}
                placeholder="Enter URL or search..."
                className="flex-1 px-3 py-2 bg-transparent text-white placeholder-gray-400 outline-none text-sm"
              />

              {/* Loading indicator or external link */}
              <div className="px-3 py-2">
                {session?.isLoading ? (
                  <RotateCcw className="w-4 h-4 animate-spin text-gray-400" />
                ) : (
                  <button
                    type="button"
                    onClick={handleExternalOpen}
                    className="p-0.5 rounded hover:bg-slate-700 transition-colors"
                    title="Open in external browser"
                  >
                    <ExternalLink className="w-3 h-3 text-gray-400" />
                  </button>
                )}
              </div>
            </div>
          </form>
        </div>

        {/* Right side - Controls */}
        <div className="flex items-center space-x-2">
          <button
            className="p-1.5 rounded-md hover:bg-slate-600 transition-colors"
            title="Bookmark"
          >
            <Star className="w-4 h-4" />
          </button>
          <button
            className="p-1.5 rounded-md hover:bg-slate-600 transition-colors"
            title="More options"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Page info bar */}
      {session && session.url !== 'about:blank' && (
        <div className="px-4 py-2 bg-slate-800 border-t border-slate-600 text-xs text-gray-400">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <span>Mode: {session.mode.toUpperCase()}</span>
              {session.title && session.title !== 'Loading...' && (
                <span>Title: {session.title}</span>
              )}
            </div>
            <div className="flex items-center space-x-2">
              {session.error && (
                <span className="text-red-400">Error: {session.error}</span>
              )}
              <span>Updated: {new Date(session.lastUpdated).toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}