import datetime as dt
import pandas as pd
import yfinance as yf
import os

TICKER = "AAPL"
START = "2025-01-01"
DATE = dt.date.today() - dt.timedelta(days=1)
END = (DATE).isoformat()

try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    base_dir = os.getcwd()

OUT_DIR = os.path.join(base_dir, "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, f"{TICKER}_{START[:4]}.csv")

def loadcsv():
    if os.path.exists(OUT_FILE):
        print(f"Data already exists in {OUT_FILE}. Loading locally...")
        return pd.read_csv(OUT_FILE)

    import requests
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    print(f"Downloading {TICKER} from {START} to {END}")
    df = yf.download(TICKER, start = START, end = END, auto_adjust = False, session=session)

    if df.empty :
        raise SystemExit("No data received")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("Columns after flatten:", list(df.columns))  # debug

    rename_map = {
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "AdjClose",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    if "AdjClose" not in df.columns and "Close" in df.columns:
        df["AdjClose"] = df["Close"]

    wanted = ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]
    present = [c for c in wanted if c in df.columns]
    df = df[present]

    subset_clean = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if subset_clean:
        df = df.dropna(subset=subset_clean)
        if "Volume" in df.columns:
            df = df[df["Volume"] > 0]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors = "coerce").dt.date.astype(str)
        df = df.sort_values("Date")

    df.to_csv(OUT_FILE, index=False)
    print(f"File saved -> {os.path.abspath(OUT_FILE)}")
    print("\nPreview:")
    print(df.head().to_string(index=False))
    print("\nInfo:")
    print(f"Rows: {len(df)} | Period: {df['Date'].min()} -> {df['Date'].max()}")
    print("Columns:", ", ".join(df.columns))

    return (df)