'use client'

import { useState, useRef, useEffect } from 'react'
import { ChevronLeft, ChevronRight, Maximize2, Minimize2 } from 'lucide-react'

interface SplitScreenLayoutProps {
  leftPanel: React.ReactNode
  rightPanel: React.ReactNode
  defaultSplit?: number // Percentage for left panel (0-100)
  minSplit?: number
  maxSplit?: number
  onSplitChange?: (leftPercentage: number) => void
  className?: string
}

export default function SplitScreenLayout({
  leftPanel,
  rightPanel,
  defaultSplit = 45,
  minSplit = 20,
  maxSplit = 80,
  onSplitChange,
  className = ''
}: SplitScreenLayoutProps) {
  const [leftPercentage, setLeftPercentage] = useState(defaultSplit)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<HTMLDivElement>(null)
  
  // Handle mouse drag for resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return
      
      const rect = containerRef.current.getBoundingClientRect()
      const newPercentage = ((e.clientX - rect.left) / rect.width) * 100
      
      // Constrain to min/max bounds
      const constrainedPercentage = Math.min(Math.max(newPercentage, minSplit), maxSplit)
      
      setLeftPercentage(constrainedPercentage)
      onSplitChange?.(constrainedPercentage)
    }
    
    const handleMouseUp = () => {
      setIsDragging(false)
    }
    
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    }
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isDragging, minSplit, maxSplit, onSplitChange])
  
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }
  
  // Quick preset functions
  const setEqualSplit = () => {
    setLeftPercentage(50)
    onSplitChange?.(50)
  }
  
  const favorLeft = () => {
    setLeftPercentage(70)
    onSplitChange?.(70)
  }
  
  const favorRight = () => {
    setLeftPercentage(30)
    onSplitChange?.(30)
  }
  
  const toggleMaximize = () => {
    if (leftPercentage > 60) {
      setLeftPercentage(30)
      onSplitChange?.(30)
    } else {
      setLeftPercentage(70)
      onSplitChange?.(70)
    }
  }
  
  return (
    <div ref={containerRef} className={`flex h-full relative ${className}`}>
      {/* Left Panel */}
      <div 
        className="flex-shrink-0 overflow-hidden"
        style={{ width: `${leftPercentage}%` }}
      >
        {leftPanel}
      </div>
      
      {/* Drag Handle */}
      <div 
        ref={dragRef}
        className={`
          relative flex-shrink-0 w-1 bg-slate-600 hover:bg-slate-500 cursor-col-resize group
          ${isDragging ? 'bg-blue-500' : ''}
        `}
        onMouseDown={handleMouseDown}
      >
        {/* Drag handle visual indicator */}
        <div className="absolute inset-y-0 left-1/2 transform -translate-x-1/2 w-3 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="w-0.5 h-8 bg-slate-400 rounded-full"></div>
        </div>
        
        {/* Control buttons - show on hover */}
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
          <div className="flex flex-col bg-slate-800 rounded-md shadow-lg border border-slate-600 overflow-hidden">
            <button
              onClick={favorLeft}
              className="p-1.5 hover:bg-slate-700 text-gray-300 hover:text-white transition-colors"
              title="Favor left panel (70%)"
            >
              <ChevronLeft className="w-3 h-3" />
            </button>
            <button
              onClick={setEqualSplit}
              className="p-1.5 hover:bg-slate-700 text-gray-300 hover:text-white transition-colors border-y border-slate-600"
              title="Equal split (50/50)"
            >
              <div className="w-3 h-3 border border-current rounded-sm"></div>
            </button>
            <button
              onClick={favorRight}
              className="p-1.5 hover:bg-slate-700 text-gray-300 hover:text-white transition-colors"
              title="Favor right panel (70%)"
            >
              <ChevronRight className="w-3 h-3" />
            </button>
            <button
              onClick={toggleMaximize}
              className="p-1.5 hover:bg-slate-700 text-gray-300 hover:text-white transition-colors border-t border-slate-600"
              title="Toggle maximize"
            >
              {leftPercentage > 60 ? (
                <Minimize2 className="w-3 h-3" />
              ) : (
                <Maximize2 className="w-3 h-3" />
              )}
            </button>
          </div>
        </div>
      </div>
      
      {/* Right Panel */}
      <div 
        className="flex-1 overflow-hidden"
        style={{ width: `${100 - leftPercentage}%` }}
      >
        {rightPanel}
      </div>
      
      {/* Split percentage indicator */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-slate-800 text-white px-2 py-1 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
        {Math.round(leftPercentage)}% | {Math.round(100 - leftPercentage)}%
      </div>
    </div>
  )
}