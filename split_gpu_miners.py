import time
from datetime import datetime
from hash_library import HashPricePredictor
from get_prices import get_latest_prices

IMMERSION_MINER_POWER = 10000
HASH_PER_MINER = 10000

def fng_to_class(fng: float) -> str:
    if fng < 20:
        return "Extreme_Fear"
    elif fng < 40:
        return "Fear"
    elif fng < 60:
        return "Neutral"
    elif fng < 80:
        return "Greed"
    else:
        return "Extreme_Greed"

def assess_once(predictor: HashPricePredictor):
    prices = get_latest_prices(30)
    latest = prices[0]
    hash_prices = [p["hash_price"] for p in reversed(prices)]

    current_hash = latest["hash_price"]
    energy_price = latest["energy_price"]

    predicted_hash = predictor.predict(
        price_window=hash_prices,
        fng_value=45.0,
        sent_class=fng_to_class(45.0),
        temp_mean=15.0,
        days_ahead=1
    )

    def profit(hp): return (hp * HASH_PER_MINER) - (energy_price * IMMERSION_MINER_POWER)

    current_profit = profit(current_hash)
    predicted_profit = profit(predicted_hash)

    result = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "current_hash_price": current_hash,
        "predicted_hash_price": predicted_hash,
        "current_profit_per_unit": current_profit,
        "predicted_profit_per_unit": predicted_profit,
        "profit_margin_watt": predicted_profit / IMMERSION_MINER_POWER,
        "is_mining_sustainable": predicted_profit > 0,
        "suggested_action": "allocate_to_mining" if predicted_profit > 0 else "reduce_mining"
    }

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Loop and print result every 5 minutes
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔁 Starting mining forecasting loop (5-min interval)...")
    predictor = HashPricePredictor("best_gru.pt", device="cpu")

    try:
        while True:
            result = assess_once(predictor)
            print(f"\n📈 [{result['timestamp']}] Mining Sustainability:")
            for k, v in result.items():
                if k != "timestamp":
                    print(f"{k:>30}: {v}")
            # Sleep until next interval (5 minutes)
            time.sleep(300)
    except KeyboardInterrupt:
        print("🛑 Stopped mining forecast loop.")
