#!/bin/bash

# MCP Router Kubernetes Deployment Script
set -e

echo "🚀 Starting MCP Router deployment..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

echo "✅ Kubernetes cluster is accessible"

# Create namespace
echo "📦 Creating namespace..."
kubectl apply -f kubernetes/namespace.yaml

# Apply ConfigMap and Secrets
echo "⚙️  Applying configuration..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml

# Deploy databases (Redis, Weaviate - SQLite is file-based)
echo "💾 Deploying databases..."
kubectl apply -f kubernetes/redis-deployment.yaml
kubectl apply -f kubernetes/weaviate-deployment.yaml

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/redis -n mcp-router
kubectl wait --for=condition=available --timeout=300s deployment/weaviate -n mcp-router

# Deploy monitoring stack
echo "📊 Deploying monitoring stack..."
kubectl apply -f kubernetes/monitoring-deployment.yaml

# Wait for monitoring to be ready
echo "⏳ Waiting for monitoring to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/jaeger -n mcp-router
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n mcp-router

# Deploy backend
echo "🔧 Deploying backend..."
kubectl apply -f kubernetes/backend-deployment.yaml

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/mcp-router-backend -n mcp-router

# Deploy frontend
echo "🌐 Deploying frontend..."
kubectl apply -f kubernetes/frontend-deployment.yaml

# Wait for frontend to be ready
echo "⏳ Waiting for frontend to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/mcp-router-frontend -n mcp-router

# Apply ingress
echo "🌍 Setting up ingress..."
kubectl apply -f kubernetes/ingress.yaml

# Display status
echo "📋 Deployment status:"
kubectl get pods -n mcp-router
echo ""
kubectl get services -n mcp-router
echo ""
kubectl get ingress -n mcp-router

echo ""
echo "🎉 MCP Router deployment completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Update your /etc/hosts file to point mcp-router.local to your cluster IP"
echo "2. Update the secrets.yaml with your actual API keys"
echo "3. Build and push your Docker images to a registry"
echo "4. Update the image references in the deployment files"
echo ""
echo "🔍 Useful commands:"
echo "  kubectl logs -f deployment/mcp-router-backend -n mcp-router"
echo "  kubectl logs -f deployment/mcp-router-frontend -n mcp-router"
echo "  kubectl port-forward service/mcp-router-frontend-service 3000:3000 -n mcp-router"
echo "  kubectl port-forward service/mcp-router-backend-service 8000:8000 -n mcp-router"
echo ""
echo "🌐 Access URLs (after setting up /etc/hosts):"
echo "  Frontend: https://mcp-router.local"
echo "  Backend API: https://mcp-router.local/api"
echo "  Jaeger UI: kubectl port-forward service/jaeger 16686:16686 -n mcp-router"
echo "  Prometheus: kubectl port-forward service/prometheus 9090:9090 -n mcp-router"