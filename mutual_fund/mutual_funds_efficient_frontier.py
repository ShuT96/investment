import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

num_port = 10000
rf = 0.01
days = 260
alpha = 0.05
tickers = ['SNP500', 'GLB_SML_CAP', 'DM_SOV', 'EM_SOV', 'JREIT']


def excel_to_df(file):
    df = pd.read_excel(file)
    mean_returns = df.pct_change().mean()
    cov = df.pct_change().cov()
    return df, mean_returns, cov


def calc_portfolio_performance(weights, mean_returns, cov, alpha, rf, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, portfolio_var, sharpe_ratio


def simulate_random_portfolios(num_portfolios, mean_returns, cov, alpha, rf, days):
    results_matrix = np.zeros((len(mean_returns) + 4, num_portfolios))
    for i in range(num_portfolios):
        # weights = [0.35, 0.02, 0.16, 0.08, 0.39]
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, portfolio_var, sharpe_ratio = calc_portfolio_performance(weights, mean_returns,
                                                                                                  cov, alpha, rf, days)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = portfolio_var
        results_matrix[3, i] = sharpe_ratio
        for j in range(len(weights)):
            results_matrix[j + 4, i] = weights[j]
    results_df = pd.DataFrame(results_matrix.T,
                              columns=['Return', 'StDev', 'VaR', 'Sharpe'] + [ticker for ticker in tickers])
    return results_df


def show_allocation_result_portfolios(results_df):
    max_return_port = results_df.iloc[results_df['Return'].idxmax()]
    min_var_port = results_df.iloc[results_df['VaR'].idxmin()]
    max_sharpe_port = results_df.iloc[results_df['Sharpe'].idxmax()]
    min_vol_port = results_df.iloc[results_df['StDev'].idxmin()]
    min_ret_port = results_df.iloc[results_df['Return'].idxmin()]

    print('-' * 80)
    print('Max Return allocation is:\n')
    print(max_return_port.to_frame().T)
    print('-' * 80)
    print('Minimum Volatility allocation is:\n')
    print(min_vol_port.to_frame().T)
    print('-' * 80)
    print('Minimum VaR allocation is:\n')
    print(min_var_port.to_frame().T)
    print('-' * 80)
    print('Max Sharpe Ratio allocation is:\n')
    print(max_sharpe_port.to_frame().T)
    print('-' * 80)
    print('Minimum Return allocation is:\n')
    print(min_ret_port.to_frame().T)

    return min_var_port, max_sharpe_port, min_vol_port, max_return_port, min_ret_port


def plot_efficient_frontier(results_df, min_VaR_port, max_sharpe_port, min_vol_port, max_return_port, min_ret_port):
    plt.subplots(figsize=(15, 10))
    plt.scatter(results_df.StDev, results_df.Return, c=results_df.Sharpe, cmap='RdYlBu')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Returns')
    plt.colorbar()

    plt.scatter(max_return_port[1], max_return_port[0], marker=(5, 1, 0), color='y', s=200)
    plt.scatter(min_VaR_port[1], min_VaR_port[0], marker=(5, 1, 0), color='b', s=200)
    plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=200)
    plt.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=200)
    plt.scatter(min_ret_port[1], min_ret_port[0], marker=(5, 1, 0), color='pink', s=200)

    plt.show()


def main():

    excel_file = r'C:\Users\shusu\Google Drive\2_Finance\2021\Rebalance\Q1.xlsx'
    pd.set_option('display.max_columns', None, 'display.expand_frame_repr', False)
    df = excel_to_df(excel_file)
    result_pf_df = simulate_random_portfolios(num_port, df[1], df[2], alpha, rf, days)
    opt_allo_df = show_allocation_result_portfolios(result_pf_df)
    plot_efficient_frontier(result_pf_df, opt_allo_df[0], opt_allo_df[1],
                            opt_allo_df[2], opt_allo_df[3], opt_allo_df[4])


if __name__ == '__main__':
    main()
