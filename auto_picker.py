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
    量化策略：【主升浪·量价突破与龙回头评分模型】(阈值自适应宽频版)
    针对震荡市放宽了成交量、回踩深度与市值的硬性门槛。
    """
    try:
        # ==========================================
        # 0. 设定时间窗口
        # ==========================================
        cal = pro.trade_cal(exchange='', is_open='1', 
                            start_date=(datetime.now() - timedelta(days=25)).strftime('%Y%m%d'), 
                            end_date=datetime.now().strftime('%Y%m%d'))
        dates_15 = cal['cal_date'].tolist()[-15:]
        dates_3 = dates_15[-3:]
        
        T0 = dates_15[-1] # 今日
        T1 = dates_15[-2] # 昨日
        
        logging.info(f"🚀 趋势接力量化引擎启动，基准日: {T0}")

        # ==========================================
        # 1. 底层行情与 MA5 计算
        # ==========================================
        df_daily_15d = pro.daily(start_date=dates_15[0], end_date=T0)
        df_daily_15d = df_daily_15d.sort_values(['ts_code', 'trade_date'])
        
        # 计算 5 日均线
        df_daily_15d['ma5'] = df_daily_15d.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        
        # 提取 T0 与 T1 行情
        df_t0 = df_daily_15d[df_daily_15d['trade_date'] == T0].copy()
        df_t1 = df_daily_15d[df_daily_15d['trade_date'] == T1][['ts_code', 'vol', 'ma5']]
        df_t1.columns = ['ts_code', 'vol_t1', 'ma5_t1']
        
        df = pd.merge(df_t0, df_t1, on='ts_code', how='inner')

        # ==========================================
        # 2. 挖掘“近期涨停”基因 (近10日内有过 >=9.5% 的涨幅)
        # ==========================================
        past_10_dates = dates_15[-11:-1] # T-10 到 T-1
        df_past_10 = df_daily_15d[df_daily_15d['trade_date'].isin(past_10_dates)]
        limit_up_codes = df_past_10[df_past_10['pct_chg'] >= 9.5]['ts_code'].unique().tolist()

        # ==========================================
        # 3. 主力资金 (Moneyflow) 获取
        # ==========================================
        df_mf_list = []
        for d in dates_3:
            try:
                mf = pro.moneyflow(trade_date=d)[['ts_code', 'net_mf_amount']]
                df_mf_list.append(mf)
            except Exception as e:
                pass
                
        if df_mf_list:
            mf_agg = pd.concat(df_mf_list).groupby('ts_code')['net_mf_amount'].sum().reset_index()
            mf_agg.rename(columns={'net_mf_amount': 'mf_3d_sum'}, inplace=True)
            df = pd.merge(df, mf_agg, on='ts_code', how='inner')
        else:
            df['mf_3d_sum'] = 0.0 

        # ==========================================
        # 4. 龙虎榜获取 (top_list)
        # ==========================================
        top_list_frames = []
        for d in dates_3:
            try:
                top_df = pro.top_list(trade_date=d)
                if not top_df.empty and 'ts_code' in top_df.columns:
                    top_list_frames.append(top_df)
            except:
                pass
            
        top_codes, top_net_buy_codes = [], []
        if top_list_frames:
            top_all = pd.concat(top_list_frames)
            top_codes = top_all['ts_code'].unique().tolist()
            if 'net_amount' in top_all.columns:
                top_net_buy_codes = top_all[top_all['net_amount'] > 0]['ts_code'].unique().tolist()

        # ==========================================
        # 5. 基础面融合
        # ==========================================
        df_basic = pro.daily_basic(trade_date=T0, fields='ts_code,turnover_rate,total_mv')
        df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        
        df = pd.merge(df, df_basic, on='ts_code', how='inner')
        df = pd.merge(df, df_info, on='ts_code', how='inner')

        # ==========================================
        # 6. 【执行适度宽频的过滤门槛】(核心调优区)
        # ==========================================
        # 法则 4：主板非ST，股价<200，换手率3%~35% (保持不变)
        cond_board = df['ts_code'].str.match(r'^(60|00)')
        cond_not_st = ~df['name'].str.contains('ST|退')
        cond_price = df['close'] < 200
        cond_turnover = (df['turnover_rate'] >= 3.0) & (df['turnover_rate'] <= 35.0)
        
        # 法则 3：量价点火 (适度放宽)
        # 涨幅1.5%起步，成交量>昨1.2倍即可，成交额>3亿(300,000千元)
        cond_ignition = (df['pct_chg'] >= 1.5) & (df['pct_chg'] <= 11.0)
        cond_volume = df['vol'] > (df['vol_t1'] * 1.2)
        cond_amt = df['amount'] >= 300000 
        
        # 法则 2：主力建仓
        cond_money = df['mf_3d_sum'] > 0

        # 法则 1：【趋势双轨承接算法】(适度放宽)
        # 轨道 A: 5日线向上 + 盘中最低价在5日线附近(放宽至1.04内，允许强势浅回踩) + 收盘站稳5日线
        cond_trend_A = (df['ma5'] > df['ma5_t1']) & (df['low'] <= df['ma5'] * 1.04) & (df['close'] > df['ma5'])
        # 轨道 B: 涨停洗盘后收复 (保持不变)
        cond_trend_B = df['ts_code'].isin(limit_up_codes) & (df['close'] > df['ma5'])
        
        cond_trend = cond_trend_A | cond_trend_B

        # 组合所有过滤网
        screened_df = df[
            cond_board & cond_not_st & cond_price & cond_turnover & 
            cond_ignition & cond_volume & cond_amt & 
            cond_money & cond_trend
        ].copy()

        if screened_df.empty:
            logging.warning("放宽阈值后今日依然无符合标的，说明市场处于极度缩量或单边下跌的冰点期。")
            return []

        # ==========================================
        # 7. 【多因子综合评分模型】(打分逻辑保持不变)
        # ==========================================
        screened_df['score'] = 0.0

        # 维度 1: 资金建仓力度
        screened_df['score'] += (screened_df['mf_3d_sum'] / 1000)
        
        # 维度 2: 资金点火爆量 (放量倍数放大 15 倍)
        screened_df['score'] += (screened_df['vol'] / screened_df['vol_t1']) * 15
        
        # 维度 3: 市场绝对核心 (成交额每多 1 亿元得 2 分)
        screened_df['score'] += (screened_df['amount'] / 100000) * 2
        
        # 维度 4: 龙虎榜光环与涨停记忆
        screened_df.loc[screened_df['ts_code'].isin(top_codes), 'score'] += 20 
        screened_df.loc[screened_df['ts_code'].isin(top_net_buy_codes), 'score'] += 15 
        screened_df.loc[screened_df['ts_code'].isin(limit_up_codes), 'score'] += 15 

        # ==========================================
        # 8. 终极决选
        # ==========================================
        top_stocks = screened_df.sort_values(by='score', ascending=False).head(5)

        stock_list = [code.split('.')[0] for code in top_stocks['ts_code'].tolist()]
        names = top_stocks['name'].tolist()
        scores = top_stocks['score'].round(1).tolist()
        
        result_display = [f"{n}({c}) 综合分:{s}" for c, n, s in zip(stock_list, names, scores)]
        logging.info(f"🏆 量化模型斩获龙头！今日前五大趋势标的: {result_display}")
        
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
