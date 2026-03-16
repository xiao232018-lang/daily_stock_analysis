import tushare as ts
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Tushare Pro
token = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(token)

def get_potential_stocks():
    """
    量化策略：【业内标准：多头趋势 + 放量突破 + 资金共振模型】
    """
    try:
        # 1. 获取最近 65 个交易日的日历，精准定位 T0, T-1, T-20, T-60
        cal = pro.trade_cal(exchange='', is_open='1', 
                            start_date=(datetime.now() - timedelta(days=100)).strftime('%Y%m%d'), 
                            end_date=datetime.now().strftime('%Y%m%d'))
        trade_dates = cal['cal_date'].tolist()
        
        T0 = trade_dates[-1]   # 今天(或最近一个交易日)
        T1 = trade_dates[-2]   # 昨天
        T20 = trade_dates[-21] # 20个交易日（约1个月前）
        T60 = trade_dates[-61] # 60个交易日（约3个月前）
        
        logging.info(f"量化引擎启动，基准日: {T0}。比对周期: {T1}, {T20}, {T60}")

        # 2. 提取不同时间节点的行情数据
        # 获取 T0 (今日) 数据
        df_t0 = pro.daily(trade_date=T0)[['ts_code', 'close', 'open', 'pct_chg', 'vol']]
        df_t0.rename(columns={'close': 'close_T0', 'vol': 'vol_T0'}, inplace=True)
        
        # 获取 T1 (昨日) 数据，用于计算是否放量
        df_t1 = pro.daily(trade_date=T1)[['ts_code', 'vol']]
        df_t1.rename(columns={'vol': 'vol_T1'}, inplace=True)
        
        # 获取 T20 和 T60 数据，用于判断长期趋势
        df_t20 = pro.daily(trade_date=T20)[['ts_code', 'close']]
        df_t20.rename(columns={'close': 'close_T20'}, inplace=True)
        
        df_t60 = pro.daily(trade_date=T60)[['ts_code', 'close']]
        df_t60.rename(columns={'close': 'close_T60'}, inplace=True)

        # 3. 获取 T0 基本面与资金面
        df_basic = pro.daily_basic(trade_date=T0, fields='ts_code,turnover_rate,total_mv')
        df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        
        df_mf = pro.moneyflow(trade_date=T0)[['ts_code', 'net_mf_amount']]

        # 4. 数据全量合并 (使用 inner join 自动剔除停牌或新股)
        df = df_t0.merge(df_t1, on='ts_code', how='inner') \
                  .merge(df_t20, on='ts_code', how='inner') \
                  .merge(df_t60, on='ts_code', how='inner') \
                  .merge(df_basic, on='ts_code', how='inner') \
                  .merge(df_info, on='ts_code', how='inner') \
                  .merge(df_mf, on='ts_code', how='inner')

        # ==========================================
        # 5. 执行科学量化筛选逻辑
        # ==========================================
        
        # A. 基础过滤：主板 + 非ST + 股价<200
        cond_board = df['ts_code'].str.match(r'^(60|00)')
        cond_not_st = ~df['name'].str.contains('ST|退')
        cond_price = df['close_T0'] < 200
        
        # B. 大势多头排列：今日收盘 > 20日 > 60日（确保不接飞刀，处于上升通道）
        cond_trend = (df['close_T0'] > df['close_T20']) & (df['close_T20'] > df['close_T60'])
        
        # C. 近期蓄势区间：20日涨幅在 5% ~ 30% 之间（剔除长期死水和已经被爆炒到高位的妖股）
        df['return_20d'] = (df['close_T0'] - df['close_T20']) / df['close_T20'] * 100
        cond_momentum = (df['return_20d'] >= 5.0) & (df['return_20d'] <= 30.0)
        
        # D. 今日点火信号：
        # 涨幅 2%~8%（温和突破，收阳线）且 成交量 > 昨日1.5倍（放量拉升，主力进场铁证）
        cond_breakout = (df['pct_chg'] >= 2.0) & (df['pct_chg'] <= 8.0) & (df['close_T0'] > df['open'])
        cond_volume = df['vol_T0'] > (df['vol_T1'] * 1.5)
        
        # E. 健康换手与主力资金护航：换手率 3~15%，且当日大单主力为净流入
        cond_turnover = (df['turnover_rate'] >= 3.0) & (df['turnover_rate'] <= 15.0)
        cond_money = df['net_mf_amount'] > 0

        # 应用所有条件
        screened_df = df[
            cond_board & cond_not_st & cond_price & 
            cond_trend & cond_momentum & 
            cond_breakout & cond_volume & 
            cond_turnover & cond_money
        ].copy()

        if screened_df.empty:
            logging.warning("今日市场环境较弱，未检测到符合【多头放量突破】的标的。")
            return []

        # 6. 优中选优：按今日主力净流入资金量从大到小排序，只取前 5
        top_stocks = screened_df.sort_values(by='net_mf_amount', ascending=False).head(5)

        # 提取股票代码并格式化
        stock_list = [code.split('.')[0] for code in top_stocks['ts_code'].tolist()]
        names = top_stocks['name'].tolist()
        
        logging.info(f"🎯 科学量化模型扫描完毕！擒获主升浪标的: {list(zip(stock_list, names))}")
        return stock_list

    except Exception as e:
        logging.error(f"量化策略运行出现异常: {e}")
        return []

if __name__ == "__main__":
    if not token:
        logging.error("未检测到 TUSHARE_TOKEN 环境变量")
    else:
        stocks = get_potential_stocks()
        if stocks:
            print(",".join(stocks))
