'use client'

// Disable static generation for error pages
export const dynamic = 'force-dynamic';

export default function NotFound() {
  return (
    <div className="flex h-screen w-screen bg-slate-900 text-white items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">404 - Page Not Found</h1>
        <p className="text-xl text-gray-300">Could not find requested resource</p>
      </div>
    </div>
  )
}