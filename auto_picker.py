import tushare as ts
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Tushare Pro
token = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(token)

def get_potential_stocks():
    """
    量化策略：【纯粹·5日线趋势与相对量能模型】
    目标：获取 4 只精准回踩或收复 5 日线、主力流入、换手活跃的趋势标的。
    """
    try:
        # ==========================================
        # 0. 设定时间窗口与日历
        # ==========================================
        cal = pro.trade_cal(exchange='', is_open='1', 
                            start_date=(datetime.now() - timedelta(days=25)).strftime('%Y%m%d'), 
                            end_date=datetime.now().strftime('%Y%m%d'))
        trade_dates = cal['cal_date'].tolist()
        
        if len(trade_dates) < 5:
            logging.warning("交易日历获取不足，请检查 Tushare 接口。")
            return []

        T0 = trade_dates[-1]  # 今日
        dates_15 = trade_dates[-15:] # 近15日用于算均线
        dates_3 = trade_dates[-3:]   # 近3日用于算资金
        
        logging.info(f"🚀 趋势极简雷达启动，基准日: {T0}")

        # ==========================================
        # 1. 行情与 MA5 相对位置计算
        # ==========================================
        df_daily = pro.daily(start_date=dates_15[0], end_date=T0)
        # 必须按时间升序排，保证 shift 平移正确
        df_daily = df_daily.sort_values(['ts_code', 'trade_date'])
        
        # 计算 5 日均线
        df_daily['ma5'] = df_daily.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        
        # 使用 shift 获取历史数据到同一行，避免复杂的 merge
        df_daily['close_t1'] = df_daily.groupby('ts_code')['close'].shift(1)
        df_daily['vol_t1'] = df_daily.groupby('ts_code')['vol'].shift(1)
        df_daily['ma5_t1'] = df_daily.groupby('ts_code')['ma5'].shift(1)
        df_daily['ma5_t3'] = df_daily.groupby('ts_code')['ma5'].shift(3)
        
        # 截取今日(T0)数据进行判定
        df_t0 = df_daily[df_daily['trade_date'] == T0].copy()

        # ==========================================
        # 2. 主力资金流入 (近3日)
        # ==========================================
        df_mf_list = []
        for d in dates_3:
            try:
                mf = pro.moneyflow(trade_date=d)[['ts_code', 'net_mf_amount']]
                if not mf.empty: df_mf_list.append(mf)
            except: pass
            
        has_mf = False
        if df_mf_list:
            mf_agg = pd.concat(df_mf_list).groupby('ts_code')['net_mf_amount'].sum().reset_index()
            mf_agg.rename(columns={'net_mf_amount': 'mf_3d_sum'}, inplace=True)
            df_t0 = pd.merge(df_t0, mf_agg, on='ts_code', how='inner')
            has_mf = True
        else:
            df_t0['mf_3d_sum'] = 0.0

        # ==========================================
        # 3. 基础面融合 (引入换手率)
        # ==========================================
        df_basic = pro.daily_basic(trade_date=T0, fields='ts_code,turnover_rate,total_mv')
        df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        
        df = pd.merge(df_t0, df_basic, on='ts_code', how='inner')
        df = pd.merge(df, df_info, on='ts_code', how='inner')

        # ==========================================
        # 4. 执行严谨的量化过滤逻辑
        # ==========================================
        
        # A. 基础：主板非ST，股价<200
        cond_board = df['ts_code'].str.match(r'^(60|00)')
        cond_not_st = ~df['name'].str.contains('ST|退')
        cond_price = df['close'] < 200
        
        # B. 相对流动性 (你提议的换手率逻辑)：
        # 换手率 >= 3.0% (代表个股自身的活跃度达标)
        # 保底成交额 >= 1亿 (100,000千元，剔除极其迷你、容易被几百万操纵的僵尸股)
        cond_liquidity = (df['turnover_rate'] >= 3.0) & (df['amount'] >= 100000)
        
        # C. 量价齐升与点火：涨幅 1% ~ 11%，量 > 昨 1.2倍
        cond_ignition = (df['pct_chg'] >= 1.0) & (df['pct_chg'] <= 11.0)
        cond_volume = df['vol'] > (df['vol_t1'] * 1.2)
        
        # D. 资金建仓：若有资金数据，近3日必须为净流入
        cond_money = (df['mf_3d_sum'] > 0) if has_mf else True
        
        # E. 神仙 5 日线算法 (回调/收复二选一)
        # 1. 均线大方向必须向上
        cond_ma5_up = df['ma5'] > df['ma5_t3']
        # 2. 今天必须收在 5 日线上
        cond_stand_on_ma5 = df['close'] > df['ma5']
        
        # 3. 灵活判定买点：
        # 情形 1 [精准回踩]: 今天盘中最低价下探，距离 5日线不到 3%，说明踩到了强支撑
        cond_pullback = df['low'] <= (df['ma5'] * 1.03)
        # 情形 2 [强势收复]: 昨天收盘还在 5日线下方(洗盘)，今天放量站回来了
        cond_recover = df['close_t1'] < df['ma5_t1']
        
        cond_trend = cond_ma5_up & cond_stand_on_ma5 & (cond_pullback | cond_recover)

        # 汇总
        screened_df = df[
            cond_board & cond_not_st & cond_price & 
            cond_liquidity & cond_ignition & cond_volume & 
            cond_money & cond_trend
        ].copy()

        if screened_df.empty:
            logging.warning("当前市场未出现完美契合【5日线回踩/收复 + 放量1.2倍】的标的。")
            return []

        # ==========================================
        # 5. 排序与精选 (取前 4 只)
        # ==========================================
        # 按【换手率】(个股相对热度) 和 【主力流入量】 降序排列
        top_stocks = screened_df.sort_values(by=['turnover_rate', 'mf_3d_sum'], ascending=[False, False]).head(4)

        stock_list = [code.split('.')[0] for code in top_stocks['ts_code'].tolist()]
        names = top_stocks['name'].tolist()
        
        logging.info(f"🎯 趋势量化雷达锁定！精选 4 只趋势标的: {list(zip(stock_list, names))}")
        
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
