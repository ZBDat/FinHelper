import akshare as ak
import pandas as pd
import datetime

def get_gold_prices():
    today = datetime.date.today().strftime("%Y%m%d")
    print(f"Fetching data for date: {today}")

    # 1. Fetch COMEX Gold Price
    # Using 'futures_foreign_commodity_realtime' or 'futures_global_commodity_hist' might be better for real-time/historical
    # Common function for foreign futures: ak.futures_foreign_commodity_realtime(subscribe_list="...")
    # But for daily data: ak.futures_foreign_hist(symbol="...")
    
    # Let's try to find the specific symbol.
    # COMEX Gold is often "GC" or "COMEX黄金" in Chinese interfaces.
    # ak.futures_foreign_hist(symbol="GC") might work.
    
    print("Fetching COMEX Gold data...")
    try:
        # Try fetching COMEX Gold futures daily data
        # Common symbols: "GC" (standard), "COMEX黄金" (Sina/Eastmoney names)
        # futures_foreign_hist is a good candidate for historical data
        # Note: Symbol might need to be specific like "GC" or "GC00Y"
        
        # Attempt 1: Try "GC"
        try:
            comex_gold_df = ak.futures_foreign_hist(symbol="GC") 
            comex_gold_df.to_csv("comex_gold.csv", index=False)
            print("Saved comex_gold.csv (Symbol: GC)")
        except:
             # Attempt 2: Try "COMEX黄金" if the above fails or returns empty
            comex_gold_df = ak.futures_foreign_hist(symbol="COMEX黄金")
            comex_gold_df.to_csv("comex_gold.csv", index=False)
            print("Saved comex_gold.csv (Symbol: COMEX黄金)")
            
    except Exception as e:
        print(f"Error fetching COMEX Gold: {e}")

    # 2. Fetch Shanghai Gold 99.99 Price
    # Symbol for Shanghai Gold 99.99 is "Au99.99"
    # Function: ak.spot_hist_sge(symbol="Au99.99") or ak.spot_sge_daily(symbol="Au99.99")
    
    print("Fetching Shanghai Gold 99.99 data...")
    try:
        # Try fetching Shanghai Gold Exchange spot price
        # recent akshare versions use spot_hist_sge
        shanghai_gold_df = ak.spot_hist_sge(symbol="Au99.99")
        shanghai_gold_df.to_csv("shanghai_gold_9999.csv", index=False)
        print("Saved shanghai_gold_9999.csv")
    except Exception as e:
        print(f"Error fetching Shanghai Gold 99.99: {e}")

if __name__ == "__main__":
    get_gold_prices()
