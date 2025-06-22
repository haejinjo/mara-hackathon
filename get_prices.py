from get_prices import get_latest_prices

recent_prices = get_latest_prices(30)
hash_prices = [p["hash_price"] for p in reversed(recent_prices)]  # oldest → newest
latest = recent_prices[0]

energy_price = latest["energy_price"]
token_price = latest["token_price"]

print(f"Energy price: {energy_price}")
print(f"Token price: {token_price}")


###################################################
from get_prices import get_latest_prices
from hash_library import HashPricePredictor

# Inventory power spec
IMMERSION_MINER_POWER = 10000  # Watts per miner
HASH_PER_MINER = 10000         # Hash units per 5 min

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

def assess_mining_sustainability():
    # Step 1: Load latest hash price from /prices
    prices = get_latest_prices(30)
    latest = prices[0]
    hash_prices = [p["hash_price"] for p in reversed(prices)]  # oldest → newest
    current_hash = latest["hash_price"]
    energy_price = latest["energy_price"]

    # Step 2: Predict next hash price
    # predictor = HashPriceForecastService()
    # predicted_hash = predictor.get_forecast(
    #     price_window=hash_prices,
    #     fng=45.0,  # static for now; replace with live FNG if available
    #     sentiment=fng_to_class(45.0),
    #     temperature=15.0,
    #     days=1
    # )

    predictor = HashPricePredictor("best_gru.pt", device="cpu")
    predicted_hash = predictor.predict(
        price_window=hash_prices,
        fng_value=45.0,
        sent_class=fng_to_class(45.0),
        temp_mean=15.0,
        days_ahead=1
    )

    # Step 3: Calculate current & predicted profit per 5 min
    def profit_per_unit(hp): return (hp * HASH_PER_MINER) - (energy_price * IMMERSION_MINER_POWER)

    current_profit = profit_per_unit(current_hash)
    predicted_profit = profit_per_unit(predicted_hash)

    profit_margin_per_watt = predicted_profit / IMMERSION_MINER_POWER

    # Step 4: Decide action
    is_sustainable = predicted_profit > 0
    action = "allocate_to_mining" if is_sustainable else "reduce_mining"

    return {
        "current_hash_price": current_hash,
        "predicted_hash_price": predicted_hash,
        "current_profit_per_unit": current_profit,
        "predicted_profit_per_unit": predicted_profit,
        "profit_margin_watt": profit_margin_per_watt,
        "is_mining_sustainable": is_sustainable,
        "suggested_action": action
    }

if __name__ == "__main__":
    result = assess_mining_sustainability()
    print("📊 Mining Sustainability Assessment:")
    for key, value in result.items():
        print(f"{key:>30}: {value}")
