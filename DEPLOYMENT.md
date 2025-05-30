# LegifAI Deployment Guide for Render

## 🚀 Quick Deployment Steps

### For Existing Render Services (Recommended)

1. **Commit and Push Changes**:
   ```bash
   git add .
   git commit -m "Add Gradio web interface to FastAPI service"
   git push origin main
   ```

2. **Update Render Service Settings**:
   - Go to your Render Dashboard
   - Select your web service
   - Navigate to **Settings**:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `cd src && python app.py`

3. **Set Environment Variables** (in Render Dashboard → Environment):
   ```
   XAI_API_KEY=your_xai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_BOE=your_pinecone_index_name_here
   OPENAI_API_KEY=your_openai_api_key_here
   LANGCHAIN_API_KEY_BOE=your_langchain_api_key_here
   LANGCHAIN_PROJECT_BOE=lawyer-ai-boe
   LANGCHAIN_TRACING_V2=true
   ```

4. **Deploy**: Click "Manual Deploy" or wait for auto-deploy

### After Deployment

Your service will be available at:
- **🌐 Web Interface**: `https://your-service.onrender.com/ui`
- **📚 API Documentation**: `https://your-service.onrender.com/docs`
- **🔗 Chat API**: `https://your-service.onrender.com/chat/invoke`

### For New Services (Using Blueprint)

1. **In Render Dashboard**: Click "New" → "Blueprint"
2. **Connect Repository**: Select your GitHub repository
3. **Auto-Configuration**: Render detects `render.yaml` and pre-fills settings
4. **Add Secret Environment Variables**: Add your API keys manually
5. **Create Service**: Click "Apply"

## 📋 Pre-Deployment Checklist

- [ ] All code committed and pushed to main branch
- [ ] `render.yaml` file is in repository root
- [ ] All API keys are ready (XAI, Pinecone, OpenAI, LangSmith)
- [ ] Build command: `pip install -r requirements.txt`
- [ ] Start command: `cd src && python app.py`
- [ ] Environment variables are set in Render dashboard

## 🔧 Troubleshooting

### Build Issues
- Check that `requirements.txt` is in the repository root
- Verify Python version compatibility (3.8+)

### Runtime Issues
- Check environment variables are set correctly
- Review logs in Render dashboard for detailed error messages
- Ensure Pinecone index exists and is accessible

### API Connection Issues
- Verify all API keys are valid and have appropriate permissions
- Check LangSmith project name matches your setup

## 📊 Monitoring

- **Health Check**: `GET /health`
- **LangSmith Traces**: Monitor in LangSmith dashboard
- **Render Logs**: Check in Render dashboard for real-time logs

## 🔄 Updates

To deploy updates:
1. Make changes locally
2. Test with `python src/test_server.py`
3. Commit and push to main branch
4. Render will auto-deploy (if enabled) or click "Manual Deploy" 