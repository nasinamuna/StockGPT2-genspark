import os
import logging
import json
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create the required directory structure for StockGPT."""
    
    # Base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    config_dir = os.path.join(base_dir, "config")
    
    # Data subdirectories
    data_raw_dir = os.path.join(data_dir, "raw")
    data_processed_dir = os.path.join(data_dir, "processed")
    data_models_dir = os.path.join(data_dir, "models")
    
    # Raw data subdirectories
    raw_market_data_dir = os.path.join(data_raw_dir, "market_data")
    raw_financial_statements_dir = os.path.join(data_raw_dir, "financial_statements")
    raw_news_social_dir = os.path.join(data_raw_dir, "news_social")
    raw_corporate_actions_dir = os.path.join(data_raw_dir, "corporate_actions")
    
    # Processed data subdirectories
    processed_market_data_dir = os.path.join(data_processed_dir, "market_data")
    processed_financial_statements_dir = os.path.join(data_processed_dir, "financial_statements")
    processed_news_social_dir = os.path.join(data_processed_dir, "news_social")
    processed_technical_indicators_dir = os.path.join(data_processed_dir, "technical_indicators")
    processed_analysis_dir = os.path.join(data_processed_dir, "analysis")
    
    # Create directories
    directories = [
        data_dir, config_dir,
        data_raw_dir, data_processed_dir, data_models_dir,
        raw_market_data_dir, raw_financial_statements_dir, raw_news_social_dir, raw_corporate_actions_dir,
        processed_market_data_dir, processed_financial_statements_dir, processed_news_social_dir,
        processed_technical_indicators_dir, processed_analysis_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "config_dir": config_dir
    }

def create_config_files(config_dir):
    """Create configuration files with default values."""
    
    # Data sources configuration
    data_sources_config = {
        "api_keys": {
            "alpha_vantage": "YOUR_ALPHA_VANTAGE_KEY",
            "news_api": "YOUR_NEWS_API_KEY",
            "twitter": "YOUR_TWITTER_API_KEY"
        },
        "market_data": {
            "sources": ["yahoo", "nse", "moneycontrol"],
            "yahoo": {
                "enabled": True
            },
            "nse": {
                "enabled": True
            },
            "moneycontrol": {
                "enabled": True,
                "mapping": {
                    "RELIANCE": "reliance-industries-ltd",
                    "TCS": "tata-consultancy-services-ltd",
                    "INFY": "infosys-ltd"
                }
            }
        },
        "financial_statements": {
            "sources": ["moneycontrol"],
            "moneycontrol": {
                "enabled": True
            }
        },
        "news_social": {
            "sources": ["news_api", "twitter"],
            "news_api": {
                "enabled": True,
                "keywords": ["stock", "market", "finance", "economy"]
            },
            "twitter": {
                "enabled": False
            }
        },
        "corporate_actions": {
            "sources": ["bse", "nse"],
            "bse": {
                "enabled": True
            },
            "nse": {
                "enabled": True
            }
        }
    }
    
    # Analysis parameters configuration
    analysis_parameters_config = {
        "technical_analysis": {
            "indicators": ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
            "sma_periods": [20, 50, 200],
            "ema_periods": [12, 26],
            "rsi_period": 14,
            "macd_parameters": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            },
            "bollinger_parameters": {
                "window": 20,
                "num_std_dev": 2
            }
        },
        "fundamental_analysis": {
            "metrics": ["P/E", "P/B", "ROE", "Debt-to-Equity", "Dividend Yield"],
            "thresholds": {
                "good_pe": 15,
                "good_pb": 3,
                "good_roe": 15,
                "good_debt_equity": 1,
                "good_dividend_yield": 3
            }
        },
        "sentiment_analysis": {
            "sources": ["news", "twitter"],
            "lookback_days": 30,
            "keywords_weight": {
                "positive": ["growth", "profit", "success", "innovation", "increase"],
                "negative": ["loss", "decline", "debt", "lawsuit", "decrease"]
            }
        },
        "risk_assessment": {
            "market_risk_weight": 0.4,
            "credit_risk_weight": 0.3,
            "geopolitical_risk_weight": 0.3
        },
        "prediction": {
            "models": ["linear", "lstm", "transformer"],
            "prediction_horizon": 30,
            "train_test_split": 0.8,
            "features": ["close", "volume", "technical_indicators", "sentiment"]
        }
    }
    
    # Stock symbols configuration
    stock_symbols_config = {
        "india": {
            "nifty50": [
                {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd.", "sector": "Energy"},
                {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd.", "sector": "Technology"},
                {"symbol": "HDFC.NS", "name": "HDFC Bank Ltd.", "sector": "Banking"},
                {"symbol": "INFY.NS", "name": "Infosys Ltd.", "sector": "Technology"},
                {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"}
            ]
        }
    }
    
    # Create config files
    config_files = {
        "data_sources.json": data_sources_config,
        "analysis_parameters.json": analysis_parameters_config,
        "stock_symbols.json": stock_symbols_config
    }
    
    for filename, config in config_files.items():
        filepath = os.path.join(config_dir, filename)
        
        # Create example file if it doesn't exist
        example_filepath = os.path.join(config_dir, f"{os.path.splitext(filename)[0]}.example.json")
        with open(example_filepath, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Created example config file: {example_filepath}")
        
        # Create actual config file if it doesn't exist
        if not os.path.exists(filepath):
            shutil.copy(example_filepath, filepath)
            logger.info(f"Created config file: {filepath}")
        else:
            logger.info(f"Config file already exists: {filepath}")
    
    return config_files

def main():
    """Set up the StockGPT project."""
    logger.info("Setting up StockGPT...")
    
    # Create directory structure
    dirs = create_directory_structure()
    
    # Create configuration files
    create_config_files(dirs["config_dir"])
    
    logger.info("Setup completed successfully!")

if __name__ == "__main__":
    main() 