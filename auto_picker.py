import akshare as ak
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_potential_stocks():
    """
    量化初筛策略：底部放量突破模型
    """
    logging.info("开始拉取 A 股全市场最新行情快照...")
    try:
        # 调用东方财富的实时/盘后快照接口，获取全市场 5000+ 股票数据
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        logging.error(f"获取全市场数据失败: {e}")
        return []

    # 1. 基础清洗：剔除退市股、ST股
    df = df[~df['名称'].astype(str).str.contains('ST|退')]
    
    # 2. 板块过滤：为了求稳，我们暂时只保留主板（60开头和00开头），剔除科创板(688)和北交所(8/4)
    df = df[df['代码'].astype(str).str.match(r'^(60|00)')]

    # 3. 核心量化逻辑筛选 (关键步骤)
    # 策略思路：寻找今天刚开始异动、有资金介入，但还没有涨停（还能买入）的股票
    condition_price_change = (df['涨跌幅'] >= 3.0) & (df['涨跌幅'] <= 8.0)  # 涨幅在 3%~8% 之间，剔除涨停买不到的
    condition_volume_ratio = df['量比'] >= 2.5  # 量比大于 2.5，说明今天成交量是过去5天的2.5倍以上，资金放量入场
    condition_turnover = (df['换手率'] >= 5.0) & (df['换手率'] <= 15.0)  # 换手率适中，交投活跃但未失控
    condition_market_cap = (df['流通市值'] >= 50_0000_0000) & (df['流通市值'] <= 300_0000_0000) # 流通市值在 50亿~300亿之间，盘子适中容易拉升

    # 应用筛选条件
    screened_df = df[condition_price_change & condition_volume_ratio & condition_turnover & condition_market_cap]

    # 4. 优中选优：按“量比”从大到小排序，提取资金攻击最猛的前 5 只
    top_stocks = screened_df.sort_values(by='量比', ascending=False).head(5)
    
    stock_list = top_stocks['代码'].tolist()
    names = top_stocks['名称'].tolist()
    
    logging.info(f"量化初筛完成，选出 {len(stock_list)} 只潜力股: {list(zip(stock_list, names))}")
    return stock_list

if __name__ == "__main__":
    stocks = get_potential_stocks()
    print(",".join(stocks))
