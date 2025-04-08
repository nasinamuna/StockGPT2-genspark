import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create placeholder classes for the components we'll use
class MarketData:
    def __init__(self):
        pass
        
    def get_stock_data(self, symbol, start_date=None, end_date=None):
        return {}
        
class TechnicalAnalysis:
    def __init__(self):
        pass
        
    def analyze(self, symbol):
        return {
            "technical_analysis": {
                "indicators": [
                    {
                        "name": "Trend",
                        "value": "Uptrend",
                        "strength": "Strong",
                        "description": "The stock is in an uptrend based on moving averages."
                    }
                ]
            }
        }
        
class FundamentalAnalysis:
    def __init__(self):
        pass
        
    def analyze(self, symbol):
        return {
            "pe_ratio": 15.5,
            "book_value": 250.0,
            "dividend_yield": 2.5,
            "debt_to_equity": 0.7,
            "summary": [
                "P/E ratio of 15.5 indicates a fairly valued stock.",
                "Book value of ₹250 suggests strong assets.",
                "Dividend yield of 2.5% is above market average.",
                "Debt-to-equity ratio of 0.7 indicates reasonable leverage.",
                "Overall fundamentals appear strong for long-term investment."
            ]
        }
        
class SentimentAnalysis:
    def __init__(self):
        pass
        
    def analyze(self, symbol):
        return {
            "sentiment": "positive",
            "score": 0.75,
            "news_summary": "Recent news indicates positive outlook due to strong quarterly results.",
            "social_media_sentiment": "Mostly positive mentions on social platforms."
        }
        
class PricePrediction:
    def __init__(self):
        pass
        
    def predict(self, symbol, days=5):
        return {
            "prediction": [100, 102, 103, 101, 105],
            "confidence": 0.8,
            "trend": "upward"
        }

# Create FastAPI app
app = FastAPI(
    title="StockGPT",
    description="An integrated platform for Indian stock market analysis",
    version="1.0.0"
)

# Setup CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directory structure if it doesn't exist
base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
data_dir = base_dir / "data"
processed_dir = data_dir / "processed"
analysis_dir = processed_dir / "analysis"

for directory in [data_dir, processed_dir, analysis_dir]:
    os.makedirs(directory, exist_ok=True)

# Initialize analysis modules
market_data = MarketData()
technical_analysis = TechnicalAnalysis()
fundamental_analysis = FundamentalAnalysis()
sentiment_analysis = SentimentAnalysis()
price_prediction = PricePrediction()

# Mock stock data
MOCK_STOCKS = {
    "RELIANCE": {"name": "Reliance Industries Ltd.", "sector": "Energy"},
    "TCS": {"name": "Tata Consultancy Services Ltd.", "sector": "Technology"},
    "HDFC": {"name": "HDFC Bank Ltd.", "sector": "Banking"},
    "INFY": {"name": "Infosys Ltd.", "sector": "Technology"},
    "SBIN": {"name": "State Bank of India", "sector": "Banking"}
}

# API Models
class StockModel(BaseModel):
    symbol: str

class StockAnalysisRequest(BaseModel):
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    analysis_type: Optional[List[str]] = ["fundamental", "technical", "sentiment"]

class StockPredictionRequest(BaseModel):
    symbol: str
    days: Optional[int] = 5

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: fastapi.Request):
    """Get list of available stocks"""
    return """
    <html>
        <head>
            <title>StockGPT API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .endpoint { margin-bottom: 20px; }
                .method { font-weight: bold; color: #008000; }
            </style>
        </head>
        <body>
            <h1>Welcome to StockGPT API</h1>
            <p>This API provides Indian stock market analysis tools including:</p>
            <ul>
                <li>Market Data Analysis</li>
                <li>Technical Analysis</li>
                <li>Fundamental Analysis</li>
                <li>Sentiment Analysis</li>
                <li>Price Prediction</li>
            </ul>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <a href="/stocks">/stocks</a></p>
                <p>Get list of available stocks</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <a href="/api/stock/{symbol}">/api/stock/{symbol}</a></p>
                <p>Get detailed stock analysis</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <a href="/api/stock/analyze">/api/stock/analyze</a></p>
                <p>Perform custom stock analysis</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <a href="/api/stock/predict">/api/stock/predict</a></p>
                <p>Get stock price prediction</p>
            </div>
            
            <p>For API documentation, visit <a href="/docs">/docs</a></p>
        </body>
    </html>
    """

@app.get("/stocks")
async def get_stocks():
    """Get list of available stocks"""
    return MOCK_STOCKS

@app.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get stock data endpoint"""
    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return {"symbol": symbol, "data": {}, "error": "Unknown symbol"}
    
    response = {
        "symbol": symbol,
        "name": MOCK_STOCKS[symbol]["name"],
        "sector": MOCK_STOCKS[symbol]["sector"],
        "info": {}
    }
    
    return response

@app.post("/api/stock/analyze")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze stock data endpoint"""
    try:
        symbol = request.symbol
        
        response = {"symbol": symbol, "data": {}}
        
        # Perform requested analyses
        if "fundamental" in request.analysis_type:
            fundamental = fundamental_analysis.analyze(symbol)
            response["data"]["fundamental"] = fundamental
            
        if "technical" in request.analysis_type:
            technical = technical_analysis.analyze(symbol)
            response["data"]["technical"] = technical
            
        if "sentiment" in request.analysis_type:
            sentiment = sentiment_analysis.analyze(symbol)
            response["data"]["sentiment"] = sentiment
        
        # Add basic stock info
        response["data"]["info"] = {
            "name": MOCK_STOCKS.get(symbol, {}).get("name", "Unknown"),
            "sector": MOCK_STOCKS.get(symbol, {}).get("sector", "Unknown"),
        }
        
        return response
    except Exception as e:
        logger.error(f"Error analyzing data for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stock/predict")
async def predict_stock(request: StockPredictionRequest):
    """Predict stock prices endpoint"""
    try:
        prediction = price_prediction.predict(request.symbol, request.days)
        return {"symbol": request.symbol, "prediction": prediction}
    except Exception as e:
        logger.error(f"Error predicting data for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Make sure app is accessible
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 