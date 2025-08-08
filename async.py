import pandas as pd
import talib
import aiohttp
import asyncio
import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from asyncio import Semaphore

# Configuration constants
BASE = "https://api.exchange.coinbase.com"
PRODUCT = "ETH-USD"
GRANULARITY = 3600 # Frequency (1 hour granularity)
DAYS = 945
CHUNK = 300 # API limit (300 minutes = 5 hours)
CONCURRENT = 10 # Concurrent requests
RETRIES = 5 # Max retry attempts
MIN = 1 # Minimum backoff time in seconds
MAX = 60 # Maximum backoff time in seconds

def timebase():
    # Returns time to the current hour
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)

# Pull request with retry logic and backoff
async def pull(session, start, end, semaphore):
    url = f"{BASE}/products/{PRODUCT}/candles"
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": GRANULARITY,
    }

    # API control with retry and backoff mechanism
    async with semaphore:
        retries = 0
        backoff = MIN
        while retries < RETRIES:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 400: # Bad Request
                        print(f"Bad request for {start} to {end}.")
                        return []
                    elif response.status == 429: # Rate limit exceeded
                        print(f"Rate limit exceeded for {start} to {end}. Retrying in {backoff} seconds...")
                        await asyncio.sleep(backoff)
                        retries += 1
                        backoff = min(backoff * 2, MAX)
                    else:
                        print(f"Received status {response.status} for {start} to {end}")
                        return []
            except Exception as e:
                retries += 1
                print(f"Exception fetching data for {start} to {end}: {e}. Retrying ({retries}/{RETRIES})...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX)
        print(f"Failed to fetch data after {RETRIES} attempts for {start} to {end}.")
        return []

# Generate chunks based on time range
async def generate(start, end):
    chunks = []
    current = start
    while current < end:
        next = min(current + timedelta(days=1), end) # Break into daily chunks
        chunks.append((current, next))
        current = next
    return chunks

# Process the data concurrently
async def process(session, chunks, semaphore):
    return await asyncio.gather(*(pull(session, s, e, semaphore) for s, e in chunks))

# Write data to CSV
async def write(filename, data):
    rows = [[datetime.fromtimestamp(entry[0], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')] + [round(entry[i], 4) for i in [3, 2, 1, 4, 5]]
        for chunk in data for entry in chunk]
    rows.sort()

    if rows:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            writer.writerows(rows)
        print("Data written to CSV.")
    else:
        print("No data parsed. Output file not written.")

async def main():
    filename = os.getenv("DATA")
    if not filename:
        raise Exception("Environment variable 'DATA' not set.")

    end = timebase()
    start = end - timedelta(days=DAYS)

    semaphore = Semaphore(CONCURRENT)

    chunks = await generate(start, end)
    async with aiohttp.ClientSession() as session:
        results = await process(session, chunks, semaphore)

    await write(filename, results)

asyncio.run(main())

def rebase():
    # Try importing TA-Lib
    try:
        import talib as ta
    except Exception as e:
        raise ImportError("TA-Lib is not properly installed. Please install the C library and Python wrapper.") from e

    # Load file from environment
    filepath = os.getenv('DATA')
    if not filepath:
        raise RuntimeError("Environment variable 'DATA' not set.")

    # Load and sort data
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    close = df['close']

    # Calculate technical indicators
    upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['lower'] = lower
    df['upper'] = upper
    df['SMA'] = middle # Use BBANDS middle band as SMA

    df['EMA'] = ta.EMA(close, timeperiod=9)

    macd, signal, hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['signal'] = signal
    df['histogram'] = hist

    df['RSI'] = ta.RSI(close, timeperiod=14)

    # Calculate VWAP (session anchored by date)
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    df['session'] = df['date'].dt.date
    tpv = hlc3 * df['volume']
    df['VWAP'] = tpv.groupby(df['session']).cumsum() / df['volume'].groupby(df['session']).cumsum()

    # Clean up
    df.dropna(inplace=True)

    # Round numeric columns
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].round(4)

    # Define column order
    base = ['date', 'open', 'high', 'low', 'close', 'volume']
    init = ['SMA', 'lower', 'upper', 'EMA', 'RSI', 'histogram', 'MACD', 'signal', 'VWAP']
    available = [col for col in init if col in df.columns]

    # Final DataFrame
    df = pd.concat([df[base], df[available]], axis=1)

    # Export
    df.to_csv(filepath, index=False)
    print("Technical indicators calculated.")

rebase()
