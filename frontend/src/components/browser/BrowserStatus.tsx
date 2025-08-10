'use client'

import { Activity, Wifi, Clock, Eye } from 'lucide-react'
import type { BrowserSession } from './BrowserShell'

interface BrowserStatusProps {
  session?: BrowserSession
  totalSessions: number
}

export default function BrowserStatus({ session, totalSessions }: BrowserStatusProps) {
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    })
  }

  const getConnectionStatus = () => {
    if (!session) return 'No connection'
    if (session.error) return 'Error'
    if (session.isLoading) return 'Connecting...'
    return 'Connected'
  }

  const getStatusColor = () => {
    if (!session) return 'text-gray-500'
    if (session.error) return 'text-red-400'
    if (session.isLoading) return 'text-yellow-400'
    return 'text-green-400'
  }

  return (
    <div className="browser-status bg-slate-800 border-t border-slate-700 px-4 py-2">
      <div className="flex items-center justify-between text-xs">
        {/* Left side - Connection status */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            <Activity className={`w-3 h-3 ${getStatusColor()}`} />
            <span className={getStatusColor()}>{getConnectionStatus()}</span>
          </div>

          {session && (
            <>
              <div className="flex items-center space-x-1 text-gray-400">
                <Eye className="w-3 h-3" />
                <span>Mode: {session.mode.toUpperCase()}</span>
              </div>

              {session.isSecure && (
                <div className="flex items-center space-x-1 text-green-400">
                  <Wifi className="w-3 h-3" />
                  <span>Secure</span>
                </div>
              )}
            </>
          )}
        </div>

        {/* Right side - Session info */}
        <div className="flex items-center space-x-4 text-gray-400">
          {session && (
            <div className="flex items-center space-x-1">
              <Clock className="w-3 h-3" />
              <span>Updated: {formatTime(session.lastUpdated)}</span>
            </div>
          )}

          <div>
            {totalSessions} session{totalSessions !== 1 ? 's' : ''}
          </div>

          <div className="text-gray-500">
            Browser Shell v1.0
          </div>
        </div>
      </div>
    </div>
  )
}