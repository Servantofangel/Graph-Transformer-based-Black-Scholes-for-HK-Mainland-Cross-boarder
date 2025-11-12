import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from simple_model import ImprovedCrossBorderGNN, generate_synthetic_market_data, create_graph_data
import scipy.stats as stats

class GraphAwareBlackScholes:
    
    def __init__(self, model, risk_free_rate=0.03):
        self.model = model
        self.r = risk_free_rate
    
    def spread_option_price(self, S_A, S_H, K, T, sigma_A, sigma_H, correlation):
        F_A = S_A * np.exp(self.r * T)
        F_H = S_H * np.exp(self.r * T)
        
        sigma_spread = np.sqrt(sigma_A**2 + sigma_H**2 - 2 * correlation * sigma_A * sigma_H)
        
        F_ratio = F_H / (F_H + K * np.exp(-self.r * T))
        sigma_adj = sigma_spread * F_ratio
        
        d1 = (np.log((F_A) / (F_H + K * np.exp(-self.r * T))) + 0.5 * sigma_adj**2 * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)
        
        price = np.exp(-self.r * T) * (F_A * stats.norm.cdf(d1) - (F_H + K * np.exp(-self.r * T)) * stats.norm.cdf(d2))
        
        return price
    
    def basket_option_price(self, S_A, S_H, K, T, sigma_A, sigma_H, correlation, weights=[0.5, 0.5]):
        S_basket = weights[0] * S_A + weights[1] * S_H
        sigma_basket = np.sqrt(
            (weights[0] * sigma_A)**2 + 
            (weights[1] * sigma_H)**2 + 
            2 * weights[0] * weights[1] * sigma_A * sigma_H * correlation
        )
        
        d1 = (np.log(S_basket / K) + (self.r + 0.5 * sigma_basket**2) * T) / (sigma_basket * np.sqrt(T))
        d2 = d1 - sigma_basket * np.sqrt(T)
        
        price = np.exp(-self.r * T) * (S_basket * np.exp(self.r * T) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        
        return price

def demonstrate_pricing_application():
    
    df = generate_synthetic_market_data(n_pairs=5, n_days=100)
    graphs = create_graph_data(df, time_steps=10)
    
    model = ImprovedCrossBorderGNN(
        node_features=8,
        edge_features=2, 
        hidden_dim=128
    )

    train_graphs = graphs[:50]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(30):
        for graph in train_graphs:
            optimizer.zero_grad()
            predictions = model.predict_correlations(graph.x, graph.edge_index, graph.edge_attr)
            target = torch.ones_like(predictions) * 0.85
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
    
    print("3. Pricing derivatives with graph-driven correlations...")
    bs_pricer = GraphAwareBlackScholes(model)
    
    latest_graph = graphs[-1]
    
    model.eval()
    with torch.no_grad():
        predicted_correlations = model.predict_correlations(
            latest_graph.x, latest_graph.edge_index, latest_graph.edge_attr
        )
    
    current_prices = {}
    for i, node_features in enumerate(latest_graph.x):
        is_A_share = node_features[3].item() > 0.5  
        price = node_features[0].item()  
        pair_id = int(node_features[7].item() * 5)  
        
        if is_A_share:
            current_prices[pair_id] = {'A': price}
        else:
            if pair_id not in current_prices:
                current_prices[pair_id] = {}
            current_prices[pair_id]['H'] = price
    
    print("Current Market State:")
    print("Pair | A-Share Price | H-Share Price | Graph Correlation")

    
    option_prices = {}
    for pair_id, prices in current_prices.items():
        if 'A' in prices and 'H' in prices:
            S_A = prices['A']
            S_H = prices['H']
            
            edge_idx = pair_id  
            if edge_idx < len(predicted_correlations):
                correlation = predicted_correlations[edge_idx].item()
            else:
                correlation = 0.8  
                
            print(f"{pair_id:4d} | {S_A:13.2f} | {S_H:13.2f} | {correlation:16.4f}")
            
            T = 0.25  
            K_spread = abs(S_A - S_H)  
            K_basket = (S_A + S_H) / 2  
            
            sigma_A = 0.25
            sigma_H = 0.30
            
            spread_price = bs_pricer.spread_option_price(
                S_A, S_H, K_spread, T, sigma_A, sigma_H, correlation
            )
            
            basket_price = bs_pricer.basket_option_price(
                S_A, S_H, K_basket, T, sigma_A, sigma_H, correlation
            )
            
            option_prices[pair_id] = {
                'spread_option': spread_price,
                'basket_option': basket_price,
                'correlation': correlation
            }
    
    print("Derivative Prices:")
    print("Pair | Spread Option | Basket Option | Correlation")
    for pair_id, prices in option_prices.items():
        print(f"{pair_id:4d} | {prices['spread_option']:13.4f} | {prices['basket_option']:12.4f} | {prices['correlation']:12.4f}")
    
    S_A_sample = 50.0
    S_H_sample = 45.0
    K = 5.0
    T = 0.25
    sigma_A = 0.25
    sigma_H = 0.30
    
    correlations = np.linspace(0.1, 0.9, 9)
    spread_prices = []
    basket_prices = []
    
    for corr in correlations:
        spread_price = bs_pricer.spread_option_price(S_A_sample, S_H_sample, K, T, sigma_A, sigma_H, corr)
        basket_price = bs_pricer.basket_option_price(S_A_sample, S_H_sample, K, T, sigma_A, sigma_H, corr)
        spread_prices.append(spread_price)
        basket_prices.append(basket_price)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(correlations, spread_prices, 'b-o', label='Spread Option', linewidth=2)
    plt.plot(correlations, basket_prices, 'r-s', label='Basket Option', linewidth=2)
    plt.xlabel('Correlation')
    plt.ylabel('Option Price')
    plt.title('Option Price vs Correlation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    pairs = list(option_prices.keys())
    spread_prices_current = [option_prices[p]['spread_option'] for p in pairs]
    basket_prices_current = [option_prices[p]['basket_option'] for p in pairs]
    
    x = np.arange(len(pairs))
    width = 0.35
    
    plt.bar(x - width/2, spread_prices_current, width, label='Spread Option', alpha=0.7)
    plt.bar(x + width/2, basket_prices_current, width, label='Basket Option', alpha=0.7)
    plt.xlabel('Stock Pair')
    plt.ylabel('Option Price')
    plt.title('Current Derivative Prices')
    plt.xticks(x, pairs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    correlations_current = [option_prices[p]['correlation'] for p in pairs]
    plt.scatter(range(len(correlations_current)), correlations_current, s=100, c=correlations_current, cmap='RdYlGn')
    plt.colorbar(label='Correlation')
    plt.xlabel('Stock Pair')
    plt.ylabel('Graph-Predicted Correlation')
    plt.title('Cross-Market Correlations')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pde_application_results.png', dpi=150, bbox_inches='tight')
    
    return model, option_prices

if __name__ == "__main__":
    model, prices = demonstrate_pricing_application()