#!/bin/bash

# Setup script for SQLite-based MCP Router
set -e

echo "🚀 Setting up SQLite-based MCP Router..."

# Create data directory for SQLite database
echo "📁 Creating data directory..."
mkdir -p backend/data
mkdir -p infrastructure/docker/data

echo "✅ Data directories created"

# Make test script executable
chmod +x backend/test_sqlite.py

echo "🧪 Running SQLite tests..."
cd backend
python test_sqlite.py

echo ""
echo "🎉 SQLite setup completed successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Set your API keys in infrastructure/docker/.env"
echo "2. Run: cd infrastructure/docker && docker-compose up -d"
echo "3. Visit: http://localhost:3000 (frontend) or http://localhost:8000/docs (API)"
echo ""
echo "💡 For Vercel deployment:"
echo "   - DATABASE_URL will automatically use SQLite"
echo "   - No external database dependencies!"
echo ""
echo "💡 For Cloudflare deployment:"
echo "   - Use D1 (managed SQLite) by updating DATABASE_URL"
echo "   - Change to: DATABASE_URL=d1://your-d1-database"