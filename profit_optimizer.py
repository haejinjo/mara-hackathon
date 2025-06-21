import numpy as np

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using simple grid search instead.")

def calculate_profit_inference(tokens, token_price, power, energy_price):
    """
    Calculate profit from inference operations.
    
    Formula: Profit = (tokens × token_price) - (power × energy_price)
    
    Args:
        tokens (float): Number of tokens processed
        token_price (float): Money made per token
        power (float): Power consumption in watts
        energy_price (float): Energy price per watt-hour
    
    Returns:
        float: Profit from inference
    """
    revenue = tokens * token_price
    cost = power * energy_price
    profit = revenue - cost
    return profit


def calculate_profit_mining(hashrate, hash_price, power, energy_price):
    """
    Calculate profit from mining operations.
    
    Formula: Profit = hashrate × hash_price - power × energy_price
    
    Args:
        hashrate (float): Mining hashrate (e.g., MH/s, GH/s)
        hash_price (float): Price per hash unit
        power (float): Power consumption in watts
        energy_price (float): Energy price per watt-hour
    
    Returns:
        float: Profit from mining
    """
    revenue = hashrate * hash_price
    cost = power * energy_price
    profit = revenue - cost
    return profit


def calculate_total_power_from_miners(miner_quantities, miner_power_consumption):
    """
    Calculate total available power based on miner quantities and their power consumption.
    
    Args:
        miner_quantities (dict): Number of each miner type {'asic_miners': int, 'gpu_compute': int, 'immersion_miners': int}
        miner_power_consumption (dict): Power consumption per miner type in watts
    
    Returns:
        float: Total available power in watts
    """
    total_power = 0
    for miner_type, quantity in miner_quantities.items():
        if miner_type in miner_power_consumption:
            total_power += quantity * miner_power_consumption[miner_type]
    return total_power


def optimize_power_allocation_with_miners(miner_quantities, miner_power_consumption, inference_params, mining_params, energy_price):
    """
    Optimize power allocation between inference and mining given specific miner quantities.
    
    Args:
        miner_quantities (dict): Number of each miner type {'asic_miners': int, 'gpu_compute': int, 'immersion_miners': int}
        miner_power_consumption (dict): Power consumption per miner type in watts
        inference_params (dict): Parameters for inference {'tokens_per_watt': float, 'token_price': float}
        mining_params (dict): Parameters for mining {'hashrate_per_watt': float, 'hash_price': float}
        energy_price (float): Energy price per watt-hour
    
    Returns:
        dict: Optimal allocation and profit breakdown
    """
    
    # Calculate total available power
    total_power = calculate_total_power_from_miners(miner_quantities, miner_power_consumption)
    
    def total_profit(power_to_inference):
        """
        Calculate total profit given power allocation to inference.
        power_to_mining = total_power - power_to_inference
        """
        if power_to_inference < 0 or power_to_inference > total_power:
            return -float('inf')  # Penalize invalid allocations
        
        power_to_mining = total_power - power_to_inference
        
        # Calculate tokens and hashrate based on power allocation
        tokens = power_to_inference * inference_params['tokens_per_watt']
        hashrate = power_to_mining * mining_params['hashrate_per_watt']
        
        # Calculate profits
        inference_profit = calculate_profit_inference(
            tokens, inference_params['token_price'], power_to_inference, energy_price
        )
        mining_profit = calculate_profit_mining(
            hashrate, mining_params['hash_price'], power_to_mining, energy_price
        )
        
        return inference_profit + mining_profit
    
    if SCIPY_AVAILABLE:
        # Use scipy optimization
        result = minimize_scalar(
            lambda x: -total_profit(x),  # Minimize negative profit = maximize profit
            bounds=(0, total_power),
            method='bounded'
        )
        optimal_inference_power = result.x
        success = result.success
    else:
        # Use simple grid search
        best_profit = -float('inf')
        optimal_inference_power = 0
        
        for power in np.linspace(0, total_power, 100):
            profit = total_profit(power)
            if profit > best_profit:
                best_profit = profit
                optimal_inference_power = power
        success = True
    
    optimal_mining_power = total_power - optimal_inference_power
    
    # Calculate final profits
    optimal_tokens = optimal_inference_power * inference_params['tokens_per_watt']
    optimal_hashrate = optimal_mining_power * mining_params['hashrate_per_watt']
    
    final_inference_profit = calculate_profit_inference(
        optimal_tokens, inference_params['token_price'], optimal_inference_power, energy_price
    )
    final_mining_profit = calculate_profit_mining(
        optimal_hashrate, mining_params['hash_price'], optimal_mining_power, energy_price
    )
    
    return {
        'total_power': total_power,
        'optimal_inference_power': optimal_inference_power,
        'optimal_mining_power': optimal_mining_power,
        'inference_profit': final_inference_profit,
        'mining_profit': final_mining_profit,
        'total_profit': final_inference_profit + final_mining_profit,
        'tokens_processed': optimal_tokens,
        'hashrate_achieved': optimal_hashrate,
        'success': success
    }


def analyze_power_allocation_with_miners(miner_quantities, miner_power_consumption, inference_params, mining_params, energy_price):
    """
    Analyze different power allocation scenarios given specific miner quantities.
    """
    total_power = calculate_total_power_from_miners(miner_quantities, miner_power_consumption)
    
    print(f"=== Power Allocation Analysis with Miners ===")
    print(f"Miner Quantities: {miner_quantities}")
    print(f"Miner Power Consumption: {miner_power_consumption}")
    print(f"Total Power Available: {total_power}W")
    print(f"Energy Price: ${energy_price}/kWh\n")
    
    # Test different allocations
    allocations = [0, 0.25, 0.5, 0.75, 1.0]  # Percentage to inference
    
    print("Allocation Scenarios:")
    print("Power to Inference | Power to Mining | Total Profit")
    print("-" * 50)
    
    best_profit = -float('inf')
    best_allocation = None
    
    for pct in allocations:
        inference_power = total_power * pct
        mining_power = total_power * (1 - pct)
        
        tokens = inference_power * inference_params['tokens_per_watt']
        hashrate = mining_power * mining_params['hashrate_per_watt']
        
        inference_profit = calculate_profit_inference(
            tokens, inference_params['token_price'], inference_power, energy_price
        )
        mining_profit = calculate_profit_mining(
            hashrate, mining_params['hash_price'], mining_power, energy_price
        )
        
        total_profit = inference_profit + mining_profit
        
        print(f"{inference_power:8.0f}W ({pct*100:3.0f}%) | {mining_power:8.0f}W ({(1-pct)*100:3.0f}%) | ${total_profit:8.2f}")
        
        if total_profit > best_profit:
            best_profit = total_profit
            best_allocation = pct
    
    if best_allocation is not None:
        print(f"\nBest allocation: {best_allocation*100:.0f}% to inference, ${best_profit:.2f} total profit")
    
    # Run optimization
    print(f"\n=== Optimization Result ===")
    optimal = optimize_power_allocation_with_miners(miner_quantities, miner_power_consumption, inference_params, mining_params, energy_price)
    
    if optimal['success']:
        print(f"Total Power: {optimal['total_power']:.1f}W")
        print(f"Optimal Inference Power: {optimal['optimal_inference_power']:.1f}W")
        print(f"Optimal Mining Power: {optimal['optimal_mining_power']:.1f}W")
        print(f"Inference Profit: ${optimal['inference_profit']:.2f}")
        print(f"Mining Profit: ${optimal['mining_profit']:.2f}")
        print(f"Total Profit: ${optimal['total_profit']:.2f}")
        print(f"Tokens Processed: {optimal['tokens_processed']:.0f}")
        print(f"Hashrate Achieved: {optimal['hashrate_achieved']:.1f} MH/s")
    else:
        print("Optimization failed!")


# Example usage
if __name__ == "__main__":
    # Example miner quantities (from your notebook)
    miner_quantities = {
        'asic_miners': 10,
        'gpu_compute': 30,
        'immersion_miners': 5
    }
    
    # Example power consumption per miner type (in watts)
    miner_power_consumption = {
        'asic_miners': 50,      # 50W per ASIC miner
        'gpu_compute': 25,      # 25W per GPU compute unit
        'immersion_miners': 100 # 100W per immersion miner
    }
    
    # Inference parameters: tokens per watt and token price
    inference_params = {
        'tokens_per_watt': 2.0,    # 2 tokens per watt
        'token_price': 0.01        # $0.01 per token
    }
    
    # Mining parameters: hashrate per watt and hash price
    mining_params = {
        'hashrate_per_watt': 0.1,  # 0.1 MH/s per watt
        'hash_price': 0.001        # $0.001 per MH/s
    }
    
    energy_price = 0.12  # $0.12 per kWh
    
    # Run analysis
    analyze_power_allocation_with_miners(miner_quantities, miner_power_consumption, inference_params, mining_params, energy_price) 