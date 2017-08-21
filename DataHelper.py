import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.optimize as spo

class DataHelper:
    @staticmethod
    def GetPriceData(symbol):
        # 01-01
        return pd.read_csv("data/{0}.csv".format(symbol))

    @staticmethod
    def GetPriceHistory(symbols, start_date='2001-01-01', end_date='2012-12-31', includeSpy = True):
        # 01-02
        dates = pd.date_range(start_date, end_date)
        if "SPY" not in symbols:
            symbols.insert(0, "SPY")
        dfRet = pd.DataFrame(index=dates)
        for s in symbols:
            dfTemp = pd.read_csv("data/{0}.csv".format(s), index_col="Date", parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
            dfTemp = dfTemp.rename(columns={'Adj Close':s})
            dfRet = dfRet.join(dfTemp)
            if s == "SPY":
                dfRet = dfRet.dropna(subset=["SPY"])
        DataHelper.fill_missing_values(dfRet)
        if(not includeSpy):
            symbols.remove("SPY")
            return dfRet.ix[:,1:]
        return dfRet
    
    @staticmethod
    def fill_missing_values(df):
        # 01-05
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='backfill', inplace=True)

    @staticmethod
    def normalize_data(df):
        # 01-02
        # normalizes everything to row 1, column 1 data point
        return df / df.ix[0,:]
    
    @staticmethod
    def plot_data(df, title="Stock prices"): 
        """Plot stock prices with a custom title and meaningful axis labels.""" 
        plt.figure(figsize=(20,10))
        ax = df.plot(title=title, fontsize=12, grid=True) 
        ax.set_xlabel("Date") 
        ax.set_ylabel("Price") 
        plt.show() 

    @staticmethod
    def get_rolling_mean(values, window):
        # 01-04
        return values.rolling(window=window).mean()

    @staticmethod
    def get_rolling_std(values, window):
        # 01-04
        return values.rolling(window=window).std()

    @staticmethod
    def get_bollinger_bands(rm, rstd):
        # 01-04
        upper_band = rm + 2 * rstd
        lower_band = rm - 2 * rstd
        return upper_band, lower_band

    @staticmethod
    def plot_bollinger_bands(data):
        # 01-04
        plt.figure(figsize=(20,10))
        rm = DataHelper.get_rolling_mean(data, 20)
        rstd = DataHelper.get_rolling_std(data, 20)
        ub, lb = DataHelper.get_bollinger_bands(rm, rstd)
        
        # Plot raw values, rolling mean and Bollinger Bands
        ax = data.plot(title="Bollinger Bands", label='SPY')
        rm.plot(label='Rolling mean', ax=ax)
        ub.plot(label='upper band', ax=ax)
        lb.plot(label='lower band', ax=ax)

        # Add axis labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')
        plt.show()

    @staticmethod
    def compute_daily_returns(df):
        # 01-04
        return ((df / df.shift(1)) - 1).fillna(0)
    
    @staticmethod
    def compute_cumulative_returns(df):
        # 01-04
        return ((df / df[0]) - 1)

    @staticmethod
    def get_alpha_beta(dfBenchmark, dfAsset):
        return np.polyfit(dfBenchmark,dfAsset,1)

    @staticmethod
    def get_beta(dfBenchmark, dfAsset):
        return DataHelper.get_alpha_beta(dfBenchmark, dfAsset)[0]

    @staticmethod
    def get_linereg(dfBenchmark, dfAsset):
        dfBenchmark = sm.add_constant(dfBenchmark)
        model = sm.OLS(dfAsset,dfBenchmark)
        return model.fit()

    @staticmethod
    def plot_linereg(dfDailyReturns, benchmarkSymbol, assetSymbol):
        dfBenchmark = dfDailyReturns[benchmarkSymbol]
        dfAsset = dfDailyReturns[assetSymbol]
        beta,alpha = DataHelper.get_alpha_beta(dfBenchmark, dfAsset)
        dfDailyReturns.plot(kind='scatter', x=benchmarkSymbol, y=assetSymbol, figsize=(10,10))
        plt.plot(dfBenchmark, beta*dfBenchmark + alpha, '-', color='r')
        plt.show()


    # optimize portfolio 
    @staticmethod
    def assess_portfolio(start_date='2010-01-01', end_date='2010-12-31', symbol_allocs = {}, start_val=1000000, rfr=0.0, sf=252.0):
        """finds the optimal alllocations for a given set of stocks using Sharpe ratio as bench mark
        
        Parameters
        ----------
        start_date: string
        end_date: string
        symbols_allocs: dictionary/OrderedDict of symbols and their allocations
        start_val: number
        
        Returns
        -------
        dictionary:
            cum_ret:
            avg_period_ret: daily by default (sf=252)
            std_dev_period_ret: daily by default (sf=252)
            sharpe_ratio:
            ending_value:
        
        """
        if(sum(symbol_allocs.values()) != 1.0):
            print("ERROR: allocations must sum to 1.0")
            return {}
        
        symbols = list(symbol_allocs.keys())
        priceHistory = DataHelper.GetPriceHistory(symbols, start_date, end_date)
        symbols = list(symbol_allocs.keys())
        data = priceHistory[symbols]
            
        # calculate cum return and port_val
        normed = DataHelper.normalize_data(data)
        alloced = normed * list(symbol_allocs.values())
        pos_vals = alloced * start_val
        port_val = pos_vals.sum(axis=1)
        cum_ret = (port_val[-1] / port_val[0]) - 1
        
        # daily returns
        daily_ret = DataHelper.compute_daily_returns(port_val)[1:,]
        
        # sharpe ratio
        std = daily_ret.std()
        sharpe_ratio = ((daily_ret.mean() - rfr) / std) * (sf **(1/2.0))
        
        return {"cum_ret": cum_ret, "daily_ret": daily_ret.mean(), "std_ret": std, "sharpe_ratio": sharpe_ratio}
        
    @staticmethod
    def get_stats(normed, weights, rfr=0.0, sf=252.0):
        alloced = normed * weights
        pos_vals = alloced * 1
        port_val = pos_vals.sum(axis=1)
        cum_ret = (port_val[-1] / port_val[0]) - 1
        daily_ret = DataHelper.compute_daily_returns(port_val)[1:,]
        mean = daily_ret.mean()
        std = daily_ret.std()
        sharpe_ratio = ((mean - rfr) / std) * (sf **(1/2.0)) 
        stats = {"cum_ret": cum_ret, "daily_ret": mean, "std_ret": std, "sharpe_ratio": sharpe_ratio}
        return stats, port_val
        
    @staticmethod
    def min_func_sharpe(weights, normed, rfr=0.0, sf=252.0):
        alloced = normed * weights
        pos_vals = alloced * 1
        port_val = pos_vals.sum(axis=1)
        daily_ret = DataHelper.compute_daily_returns(port_val)[1:,]
        std = daily_ret.std()
        sharpe_ratio = ((daily_ret.mean() - rfr) / std) * (sf **(1/2.0))
        return -sharpe_ratio

    @staticmethod
    def optimize_portfolio(start_date, end_date, symbols, gen_plot=False):
        noa = len(symbols) # number of assets
        priceHistory = DataHelper.GetPriceHistory(symbols, start_date, end_date, False)
        normed = DataHelper.normalize_data(priceHistory)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0,1) for x in range(noa))
        
        initial_guess = noa * [1. / noa]
        
        opts = spo.minimize(DataHelper.min_func_sharpe, initial_guess, args=(normed,), method='SLSQP', bounds=bnds, constraints=cons )
        
        ret,pval = DataHelper.get_stats(normed, opts.x)
        ret["optimal_allocs"] = opts.x
        ret["SPY_stats"] = DataHelper.assess_portfolio(start_date, end_date, symbol_allocs = {'SPY': 1.0})
        
        if(gen_plot):
            
            spyHistory = DataHelper.GetPriceHistory(['SPY'], start_date, end_date)
            spyNorm = DataHelper.normalize_data(spyHistory)
            
            ax = spyNorm.plot(label="SPY", figsize=(20,10))
            plt.plot(pval, label='Portfolio')
            plt.legend(loc=0)
            
            plt.show()
        
        return ret