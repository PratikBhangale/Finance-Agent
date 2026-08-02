import yfinance as yf
import pandas as pd
from typing import Dict, Any

def get_historical_market_data(ticker_symbol: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
    """
    Fetches historical market data for a given stock ticker.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        period (str): The time period for historical data. Valid values include: 
                      '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'. Default is '1mo'.
        interval (str): The frequency of the data. Valid values include: 
                        '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'. Default is '1d'.

    Returns:
        Dict[str, Any]: A dictionary containing the historical price data in a list-of-records format, or an error message.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        history_df = ticker.history(period=period, interval=interval)
        
        if history_df.empty:
            return {"error": f"No historical data found for {ticker_symbol} with period '{period}' and interval '{interval}'."}
            
        # Reset index to include 'Date' or 'Datetime' as a standard dictionary key
        # Format as records so the LLM gets a clean, readable JSON array
        return {"data": history_df.reset_index().to_dict(orient="records")}
        
    except Exception as e:
        return {"error": f"Failed to retrieve market data: {str(e)}"}


def get_fundamental_data(ticker_symbol: str, statement_type: str = "income") -> Dict[str, Any]:
    """
    Retrieves fundamental financial statements (Income Statement, Balance Sheet, or Cash Flow) for a company.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'TSLA').
        statement_type (str): The type of financial statement to retrieve. 
                              Must be 'income', 'balance_sheet', or 'cash_flow'. Default is 'income'.

    Returns:
        Dict[str, Any]: A dictionary representation of the requested financial statement organized by date.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        if statement_type == "income":
            data_df = ticker.financials
        elif statement_type == "balance_sheet":
            data_df = ticker.balance_sheet
        elif statement_type == "cash_flow":
            data_df = ticker.cashflow
        else:
            return {"error": "Invalid statement_type. Must be 'income', 'balance_sheet', or 'cash_flow'."}
            
        if data_df is None or data_df.empty:
            return {"error": f"No {statement_type} data available for {ticker_symbol}."}
            
        # Convert the dates (columns) to strings and fill NaNs to prevent parsing errors
        data_df.columns = data_df.columns.astype(str)
        return {"data": data_df.fillna("N/A").to_dict()}
        
    except Exception as e:
        return {"error": f"Failed to retrieve fundamental data: {str(e)}"}


def get_corporate_actions(ticker_symbol: str) -> Dict[str, Any]:
    """
    Retrieves a chronological history of corporate actions, specifically dividends and stock splits.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AMZN').

    Returns:
        Dict[str, Any]: A dictionary containing a list of recorded corporate events.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        actions_df = ticker.actions
        
        if actions_df is None or actions_df.empty:
            return {"message": f"No corporate actions found on record for {ticker_symbol}."}
            
        return {"data": actions_df.reset_index().to_dict(orient="records")}
        
    except Exception as e:
        return {"error": f"Failed to retrieve corporate actions: {str(e)}"}


def get_company_info(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetches comprehensive metadata and descriptive information about a company.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'NVDA').

    Returns:
        Dict[str, Any]: A dictionary containing company profile, sector, summary, market cap, and key ratios.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        if not info:
            return {"error": f"No company information available for {ticker_symbol}."}
            
        # You can optionally filter this dictionary down if context window limits are a concern.
        # Returning standard critical info:
        return {
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A")
        }
        
    except Exception as e:
        return {"error": f"Failed to retrieve company metadata: {str(e)}"}