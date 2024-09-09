import numpy as np
import pandas as pd
import empyrical as ep

returns = np.array([0.01, 0.02, -0.03, 0.04, -0.02])
cum_returns = ep.cum_returns(returns)
print("Retorno acumulado:", cum_returns)

max_dd = ep.max_drawdown(returns)
print("Máxima caída:", max_dd)

benchmark_returns = np.array([0.02, 0.01, 0.03, 0.01, -0.01])
alpha, beta = ep.alpha_beta(returns, benchmark_returns)
print("Alpha:", alpha, "Beta:", beta)

sharpe_ratio = ep.sharpe_ratio(returns)
print("Ratio de Sharpe:", sharpe_ratio)

returns = pd.Series([0.01, 0.02, -0.03, 0.04, -0.02])
rolling_sharpe = ep.roll_sharpe_ratio(returns, window=3)
print("Sharpe Ratio rodante:", rolling_sharpe)

