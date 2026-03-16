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
    量化策略：【极简·5日线稳健趋势与放量点火模型】
    目标：获取 4 只完美踩稳5日线、量价齐升且主力建仓的纯正趋势股。
    """
    try:
        # ==========================================
        # 0. 设定时间窗口与日历
        # ==========================================
        cal = pro.trade_cal(exchange='', is_open='1', 
                            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), 
                            end_date=datetime.now().strftime('%Y%m%d'))
        trade_dates = cal['cal_date'].tolist()
        
        T0 = trade_dates[-1]  # 今日
        T1 = trade_dates[-2]  # 昨日
        T2 = trade_dates[-3]  # 前日
        T3 = trade_dates[-4]  # 大前天
        
        dates_15 = trade_dates[-15:] # 取近15日计算MA5
        dates_3 = trade_dates[-3:]   # 近3日看主力资金
        
        logging.info(f"🚀 极简趋势雷达启动，基准日: {T0}")

        # ==========================================
        # 1. 底层行情、MA5 计算与时间平移
        # ==========================================
        df_daily = pro.daily(start_date=dates_15[0], end_date=T0)
        df_daily = df_daily.sort_values(['ts_code', 'trade_date'])
        
        # 计算 5 日均线
        df_daily['ma5'] = df_daily.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        
        # 将 T1, T2, T3 的关键数据平移到 T0 行，方便直接做对比
        df_daily['close_t1'] = df_daily.groupby('ts_code')['close'].shift(1)
        df_daily['ma5_t1'] = df_daily.groupby('ts_code')['ma5'].shift(1)
        df_daily['vol_t1'] = df_daily.groupby('ts_code')['vol'].shift(1)
        
        df_daily['close_t2'] = df_daily.groupby('ts_code')['close'].shift(2)
        df_daily['ma5_t2'] = df_daily.groupby('ts_code')['ma5'].shift(2)
        
        df_daily['ma5_t3'] = df_daily.groupby('ts_code')['ma5'].shift(3)
        
        # 只保留 T0 (今日) 的切片数据
        df_t0 = df_daily[df_daily['trade_date'] == T0].copy()

        # ==========================================
        # 2. 主力资金持续流入建仓 (近3日)
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
        # 3. 基础面融合
        # ==========================================
        df_basic = pro.daily_basic(trade_date=T0, fields='ts_code,turnover_rate,total_mv')
        df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        
        df = pd.merge(df_t0, df_basic, on='ts_code', how='inner')
        df = pd.merge(df, df_info, on='ts_code', how='inner')

        # ==========================================
        # 4. 执行严谨的量化过滤逻辑
        # ==========================================
        
        # A. 基础要求：主板非ST，股价<200，换手活跃
        cond_board = df['ts_code'].str.match(r'^(60|00)')
        cond_not_st = ~df['name'].str.contains('ST|退')
        cond_price = df['close'] < 200
        cond_turnover = df['turnover_rate'] >= 3.0
        
        # B. 量价齐升与流动性：
        # 今日涨幅 1% ~ 11%
        # 成交量 > 昨 1.2倍
        # 成交额 > 5亿 (Tushare单位千元，500,000=5亿)
        cond_ignition = (df['pct_chg'] >= 1.0) & (df['pct_chg'] <= 11.0)
        cond_volume = df['vol'] > (df['vol_t1'] * 1.2)
        cond_amt = df['amount'] >= 500000 
        
        # C. 资金建仓：
        # 如果获取到了资金数据，要求近3天主力累计净流入 > 0
        cond_money = (df['mf_3d_sum'] > 0) if has_mf else True
        
        # D. 核心趋势：神仙 5 日线算法
        # 1. 稳步上涨: 今天的5日线必须高于大前天的5日线 (ma5 > ma5_t3)
        # 2. 今日站稳: 今天的收盘价必须 > 今天的5日线
        # 3. 隔日收复(拒绝连续破位): 昨天或前天，必须有至少一天是站稳在当时5日线之上的
        cond_trend = (
            (df['ma5'] > df['ma5_t3']) & 
            (df['close'] > df['ma5']) & 
            ((df['close_t1'] > df['ma5_t1']) | (df['close_t2'] > df['ma5_t2']))
        )

        # 汇总所有条件
        screened_df = df[
            cond_board & cond_not_st & cond_price & cond_turnover & 
            cond_ignition & cond_volume & cond_amt & 
            cond_money & cond_trend
        ].copy()

        if screened_df.empty:
            logging.warning("今日市场环境较弱，未捕捉到完美贴合【5日线强承接+爆量】的标的。")
            return []

        # ==========================================
        # 5. 排序与精选 (取前 4 只)
        # ==========================================
        # 按【今日成交额】(代表市场绝对聚焦人气) 和 【主力流入量】 降序排列
        top_stocks = screened_df.sort_values(by=['amount', 'mf_3d_sum'], ascending=[False, False]).head(4)

        stock_list = [code.split('.')[0] for code in top_stocks['ts_code'].tolist()]
        names = top_stocks['name'].tolist()
        
        logging.info(f"🎯 趋势量化模型斩获龙头！精选 4 只标的: {list(zip(stock_list, names))}")
        
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
