'use client'

import { useState, useEffect } from 'react'
import StreamingChatInterface from '@/components/StreamingChatInterface'
import { v4 as uuidv4 } from 'uuid'

// Disable static generation for this page
export const dynamic = 'force-dynamic';

export default function Home() {
  const [sessionId, setSessionId] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Generate or retrieve session ID
    let storedSessionId = localStorage.getItem('mcp_session_id')
    if (!storedSessionId) {
      storedSessionId = uuidv4()
      localStorage.setItem('mcp_session_id', storedSessionId)
    }
    setSessionId(storedSessionId)
    setIsLoading(false)
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-screen w-screen bg-slate-900 text-white items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">MCP Router</h1>
          <p className="text-xl text-gray-300 mb-8">AI Agent Interface Loading...</p>
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-white mx-auto"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen w-screen bg-slate-900 text-white">
      <StreamingChatInterface sessionId={sessionId} />
    </div>
  )
}
