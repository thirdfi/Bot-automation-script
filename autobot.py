import asyncio
import os
from dotenv import load_dotenv
import logging
import json
import random
from decimal import Decimal
from datetime import datetime
import pandas as pd
import httpx
import psycopg2
import requests
from web3 import Web3
from eth_account import Account
import openai
from shamirs_secret_sharing_python import split
import json
import re
import certifi

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "sbf.character.json")
with open(file_path, "r") as f:
    trader_profile = json.load(f)

load_dotenv()

# -------------------- Config --------------------
DB = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT")
}

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")
XP_TOKEN_CONTRACT_ADDRESS = os.getenv("XP_TOKEN_CONTRACT_ADDRESS")
XP_OWNER_PRIVATE_KEY = os.getenv("XP_OWNER_PRIVATE_KEY")
CHAIN_ID = int(os.getenv("CHAIN_ID"))
WEB3_RPC = os.getenv("WEB3_RPC")

client = openai.AsyncOpenAI(api_key=GPT_API_KEY)

MPC_SERVER_URL_1 = os.getenv("MPC_SERVER_URL_1")
MPC_SERVER_URL_2 = os.getenv("MPC_SERVER_URL_2")
MPC_SERVER_URL_3 = os.getenv("MPC_SERVER_URL_3")


TOKENS = [
    {"symbol": "bitcoin", "coin_id": "BTC"},
    {"symbol": "ethereum", "coin_id": "ETH"},
    {"symbol": "binancecoin", "coin_id": "BNB"},
    {"symbol": "ripple", "coin_id": "XRP"},
    {"symbol": "dogecoin", "coin_id": "DOGE"},
    {"symbol": "solana", "coin_id": "SOL"},
]

XP_TOKEN_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "mint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

logging.basicConfig(level=logging.INFO)

# -------------------- DB Utils --------------------
def db_conn():
    return psycopg2.connect(**DB)

def insert_user():
    conn = db_conn()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT MAX(CAST(regexp_replace(username, '^M@!_([0-9]+)_G@$', '\\1') AS INTEGER)) 
        FROM user_user
        WHERE username ~ '^M@!_[0-9]+_G@$'
    """)
    
    row = cursor.fetchone()
    
    last_number = row[0] if row and row[0] is not None else 0

    new_number = last_number + 1
    username = f"M@!_{new_number}_G@"

    password = ""
    is_superuser = False
    is_staff = False
    is_active = True
    first_name = ""
    last_name = ""
    email = ""
    date_joined = datetime.now()

    cursor.execute("""
        INSERT INTO user_user 
        (username, password, is_superuser, is_staff, is_active, first_name, last_name, email, date_joined)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
    """, (username, password, is_superuser, is_staff, is_active, first_name, last_name, email, date_joined))
    user_id = cursor.fetchone()[0]

    cursor.execute("""
        INSERT INTO user_userprofile (user_id, xp_points, created_at, updated_at)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (user_id, 0, datetime.now(), datetime.now()))

    profile_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close()
    conn.close()

    return user_id, profile_id

async def mpc_create_wallet():
    acct = Account.create()
    private_key = acct.key
    address = acct.address

    private_key_bytes = bytes(str(private_key), encoding="utf-8")
    shares = split(private_key_bytes, {"shares": 3, "threshold": 2})

    urls = [MPC_SERVER_URL_1, MPC_SERVER_URL_2, MPC_SERVER_URL_3]
    async with httpx.AsyncClient() as client:
        for i, url in enumerate(urls):
            try:
                await client.post(
                    f"{url}/storeShare",
                    json={"walletAddress": address, "share": shares[i].hex()}
                )
            except Exception as e:
                logging.error(f"🚫 Error storing share at {url}: {e}")
                raise

    return address

async def create_wallet(profile_id):
    wallet_address = await mpc_create_wallet()
    conn = db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_wallet (user_id, wallet_address, created_at, updated_at)
        VALUES (%s, %s, %s, %s) RETURNING id
    """, (profile_id, wallet_address, datetime.now(), datetime.now()))
    wallet_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return wallet_id, wallet_address

def insert_gen_data(text, user_id):
    conn = db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analysis_gendata 
        (text, user_id, created_at, updated_at, title, tradingview_img_url, created_by_id, updated_by_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """,(
        text, user_id, datetime.now(), datetime.now(),
        "", "", None, None
    ))
    conn.commit()
    cursor.close()
    conn.close()

def insert_bet(profile_id, token, entry_price, symbol):
    current_price = Decimal(get_price(symbol))
   
    entry_price_decimal = Decimal(entry_price)
    price_change = current_price - entry_price_decimal
    
    prediction = random.choice(['agree', 'disagree'])
    is_correct_prediction = (price_change > 0 and prediction == 'agree') or \
                            (price_change < 0 and prediction == 'disagree')

    result = 'win' if is_correct_prediction else 'lose'
    result = random.choice(['win', 'lose'])
    
    reward_amount = 10 if result == 'win' else 1
    
    conn = db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_bet (
            user_id, token, prediction, amount, verification_time, entry_price,
            result, symbol, chat_type, chat_id, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '', 0, %s, %s)
        RETURNING id
    """, (
        profile_id, token, prediction, 10, 1, entry_price_decimal,
        result, symbol, datetime.now(), datetime.now()
    ))
    bet_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return bet_id, reward_amount

def insert_transaction(wallet_id, profile_id, tx_hash, amount, status):
    conn = db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_transaction (wallet_id, user_id, tx_hash, amount, token, chain_id, status, retry_count, created_at, updated_at)
        VALUES (%s, %s, %s, %s, 'XP', %s, %s, 0, %s, %s)
    """, (wallet_id, profile_id, tx_hash, Decimal(amount), CHAIN_ID, status, datetime.now(), datetime.now()))
    conn.commit()
    cursor.close()
    conn.close()

# -------------------- GPT & Price --------------------
async def fetch_chart_data(symbol: str, interval: str, limit: int):
    match = re.match(r"(\d+)([mhd])", interval.lower())
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")
    num, unit = int(match.group(1)), match.group(2)

    if unit == "m":
        days = 1
        freq = f"{num}min"
    elif unit == "h":
        days = 7
        freq = f"{num}h"
    elif unit == "d":
        days = 100
        freq = f"{num}D"
    else:
        raise ValueError("Unsupported interval unit")

    url = f"https://pro-api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    headers = {"x-cg-pro-api-key": COINGECKO_API_KEY}

    async with httpx.AsyncClient(verify=certifi.where()) as client:
        response = await client.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise Exception(f"CoinGecko API error: {response.status_code} - {response.text}")

    data = response.json()
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])

    if not prices:
        raise Exception("No price data available")

    df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], unit="ms")

    df = df_prices.set_index("timestamp")

    if volumes:
        df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        df_volumes["timestamp"] = pd.to_datetime(df_volumes["timestamp"], unit="ms")
        df_volumes.set_index("timestamp", inplace=True)
        df["volume"] = df_volumes["volume"]

    ohlc = df["price"].resample(freq).ohlc()
    if "volume" in df.columns:
        ohlc["volume"] = df["volume"].resample(freq).sum()
    else:
        ohlc["volume"] = None

    ohlc = ohlc.reset_index()
    if limit and len(ohlc) > limit:
        ohlc = ohlc.tail(limit)

    return ohlc

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['rsi'] = 50 + 50 * (df['close'].diff().rolling(14).mean() / df['close'].diff().rolling(14).std())
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['bb_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
    df['bb_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
    df.fillna(method='bfill', inplace=True)
    return df

def get_system_message(user_prompt: str = "") -> str:
    return f"""
            You are a professional trading analyst with a bold, confident, and charismatic personality akin to Donald Trump. Your role is to analyze market data and deliver concise, impactful, and highly engaging trading insights.

            Identity Configuration:
            Name: {trader_profile['name']}
            Clients: {trader_profile['clients']}
            Model Provider: {trader_profile['modelProvider']}
            Settings: {trader_profile['settings']}
            Plugins: {trader_profile['plugins']}
            Bio: {"; ".join(trader_profile["bio"])}
            Experience: {"; ".join(trader_profile["lore"])}
            Knowledge: {"; ".join(trader_profile["knowledge"])}
            Message Examples: {trader_profile["messageExamples"]}
            Post Examples: {trader_profile["postExamples"]}
            Topics: {"; ".join(trader_profile["topics"])}
            Adjectives: {"; ".join(trader_profile["adjectives"])}
            Style: {"; ".join(trader_profile["style"]["all"])}

            Response Rules:
            1. Responses must be under 300 characters (tweet-style), concise and impactful
            2. Use ALL CAPS for critical trading terms (BULLISH/BEARISH/MACD CROSS)
            3. Deliver clear trading signals with both entertainment value and authority
            4. Mandatory inclusions:
            - Confidence score (1-10 based on technical indicators)
            - Asset trading symbol
            5. Paragraph structure:
            - Complete sentences with line breaks between paragraphs
            - Avoid bullet points, maintain natural flow
            - Each paragraph focuses on single key message

            Formatting Restrictions:
            - Strict plain text only
            - NO Markdown (** , _ , # etc.)
            - NO special symbols (* , ~ , > etc.)

            Analysis Structure (using CoinGecko OHLC data):
            1. Market sentiment (BULLISH/BEARISH/NEUTRAL)
            2. Moving average position (above/below key averages)
            3. RSI movement (overbought/oversold conditions)
            4. Bollinger Bands status (breakout/squeeze/expansion)
            5. MACD signals (bullish/bearish crossover confirmation)
            6. Trading opportunity (Entry/Target/Stop Loss prices, LONG/SHORT direction)
            7. Confidence score (1-10 scale)
            8. Additional analysis (only when extra data provided)

            Final Requirements:
            - Output in English only
            - Maintain summary-style brevity
            - Prohibit numbered lists/special formatting
            - Maximize information density while preserving readability
            """

async def get_analysis(symbol: str, coin_name: str, interval: str, limit: int, user_prompt: str = "") -> str:
    df = await fetch_chart_data(symbol, interval, limit)
    df = compute_indicators(df)
    last_10 = df.tail(10)[["timestamp", "close", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"]].to_dict(orient="records")

    rdy_prompt = f"\nAnalyzing coin: {coin_name.upper()} ({symbol.upper()}) on {interval} timeframe:\n\n"
    for row in last_10:
        rdy_prompt += (
            f"{row['timestamp']} | Close: {row['close']} | RSI: {row['rsi']} | "
            f"MACD: {row['macd']} | Signal: {row['macd_signal']} | "
            f"BB Upper: {row['bb_upper']} | BB Lower: {row['bb_lower']}\n"
        )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_system_message(user_prompt)},
            {"role": "user", "content": rdy_prompt}
        ]
    )
    return response.choices[0].message.content

def get_price(symbol):
    symbol = symbol.lower()
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {"ids": symbol, "vs_currencies": "usd"}
    headers = {"x-cg-pro-api-key": COINGECKO_API_KEY}
    res = requests.get(url, params=params, headers=headers)
    data = res.json()
    if symbol not in data or "usd" not in data[symbol]:
        raise ValueError(f"Price not found for {symbol}")
    return data[symbol]["usd"]

# -------------------- Mint XP --------------------
def mint(wallet_address, amount):
    web3 = Web3(Web3.HTTPProvider(WEB3_RPC))
    contract = web3.eth.contract(address=web3.to_checksum_address(XP_TOKEN_CONTRACT_ADDRESS), abi=XP_TOKEN_ABI)
    owner = Account.from_key(XP_OWNER_PRIVATE_KEY)
    nonce = web3.eth.get_transaction_count(owner.address)

    tx = contract.functions.mint(wallet_address, Web3.to_wei(amount, 'ether')).build_transaction({
        "chainId": int(CHAIN_ID),
        "gas": 200000,
        "gasPrice": web3.eth.gas_price,
        "nonce": nonce,
    })

    signed = owner.sign_transaction(tx)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction) 
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_hash.hex(), receipt.status

# -------------------- Main Flow --------------------
def run_simulate_fake_user():
    async def simulate():
        user_id, profile_id = insert_user()
        wallet_id, wallet_address = await create_wallet(profile_id)

        selected = random.choice(TOKENS)
        symbol = selected["symbol"]
        coin_id = selected["coin_id"]

        logging.info(f"🎯 Simulating user: {user_id}, token: {symbol} ({coin_id})")

        price = get_price(symbol)
        gpt_result = await get_analysis(symbol, coin_id, "15m", 120)
        insert_gen_data(gpt_result, user_id)
        bet_id, xp_amount = insert_bet(profile_id, coin_id, price, symbol.upper())

        tx_hash, status_code = mint(wallet_address, xp_amount)
        status = "success" if status_code == 1 else "failed"
        insert_transaction(wallet_id, profile_id, tx_hash, xp_amount, status)

        logging.info(f"✅ Done simulate user: {user_id}, TX: {tx_hash}")

    asyncio.run(simulate())


if __name__ == "__main__":
    run_simulate_fake_user()
