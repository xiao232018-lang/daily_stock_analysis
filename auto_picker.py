import tushare as ts
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

token = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(token)

def get_potential_stocks():
    """
    量化策略：【主升浪·双轨猎手模型】(高容错版)
    轨道一（3只）：稳健趋势核心（踩5日线、量价突破、主力流入）
    轨道二（2只）：情绪龙头反包（近期涨停、龙虎榜、洗盘后收复5日线）
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
        
        dates_15 = trade_dates[-15:]
        dates_5 = trade_dates[-5:] # 近一周 (龙虎榜)
        dates_3 = trade_dates[-3:] # 近3日 (资金)
        
        logging.info(f"🚀 双轨猎手引擎启动，基准日: {T0}")

        # ==========================================
        # 1. 底层行情与 MA5 计算
        # ==========================================
        df_daily = pro.daily(start_date=dates_15[0], end_date=T0)
        df_daily = df_daily.sort_values(['ts_code', 'trade_date'])
        
        # 计算 5 日均线
        df_daily['ma5'] = df_daily.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        
        df_t0 = df_daily[df_daily['trade_date'] == T0].copy()
        df_t1 = df_daily[df_daily['trade_date'] == T1][['ts_code', 'close', 'vol', 'ma5']].rename(columns={'close': 'close_t1', 'vol': 'vol_t1', 'ma5': 'ma5_t1'})
        df_t2 = df_daily[df_daily['trade_date'] == T2][['ts_code', 'close', 'ma5']].rename(columns={'close': 'close_t2', 'ma5': 'ma5_t2'})
        
        df = df_t0.merge(df_t1, on='ts_code', how='inner').merge(df_t2, on='ts_code', how='inner')

        # ==========================================
        # 2. 涨停基因 (近10日)
        # ==========================================
        past_10_dates = dates_15[-11:-1]
        df_past_10 = df_daily[df_daily['trade_date'].isin(past_10_dates)]
        limit_up_codes = df_past_10[df_past_10['pct_chg'] >= 9.5]['ts_code'].unique().tolist()

        # ==========================================
        # 3. 主力资金 (近3日) 与 龙虎榜 (近5日)
        # ==========================================
        # 资金防爆破处理
        df_mf_list = []
        for d in dates_3:
            try:
                mf = pro.moneyflow(trade_date=d)[['ts_code', 'net_mf_amount']]
                if not mf.empty: df_mf_list.append(mf)
            except: pass
            
        has_mf_data = False
        if df_mf_list:
            mf_agg = pd.concat(df_mf_list).groupby('ts_code')['net_mf_amount'].sum().reset_index()
            mf_agg.rename(columns={'net_mf_amount': 'mf_3d_sum'}, inplace=True)
            df = pd.merge(df, mf_agg, on='ts_code', how='inner')
            has_mf_data = True
        else:
            df['mf_3d_sum'] = 0.0

        # 龙虎榜
        top_codes_5d = []
        for d in dates_5:
            try:
                top_df = pro.top_list(trade_date=d)
                if not top_df.empty and 'ts_code' in top_df.columns:
                    top_codes_5d.extend(top_df['ts_code'].tolist())
            except: pass
        top_codes_5d = list(set(top_codes_5d))

        # ==========================================
        # 4. 基础面融合
        # ==========================================
        df_basic = pro.daily_basic(trade_date=T0, fields='ts_code,turnover_rate,total_mv')
        df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        df = pd.merge(df, df_basic, on='ts_code', how='inner')
        df = pd.merge(df, df_info, on='ts_code', how='inner')

        # ==========================================
        # 5. 全局基础硬门槛 (活跃度与流动性)
        # ==========================================
        cond_base = (
            df['ts_code'].str.match(r'^(60|00)') & 
            (~df['name'].str.contains('ST|退')) & 
            (df['close'] < 200) & 
            (df['turnover_rate'] >= 3.0) & 
            (df['amount'] >= 500000) # 5亿成交额
        )
        df_base = df[cond_base].copy()

        # ==========================================
        # 策略一：【稳健趋势股】(抽取 3 只)
        # ==========================================
        # 1. 量价齐升：今日涨幅 1%~11%，量 > 昨 1.2倍
        cond_A_ignition = (df_base['pct_chg'] >= 1.0) & (df_base['pct_chg'] <= 11.0)
        cond_A_volume = df_base['vol'] > (df_base['vol_t1'] * 1.2)
        
        # 2. 主力建仓：如果有资金数据，则要求 > 0；如果 Tushare 没数据，跳过该限制防全军覆没
        cond_A_money = (df_base['mf_3d_sum'] > 0) if has_mf_data else True
        
        # 3. 沿5日线稳步上涨算法 (极简暴力逻辑)：
        # A. 今日必须收在 5 日线上 (close > ma5)
        # B. 5 日线必须是向上的 (ma5 >= ma5_t1)
        # C. 决不连续两天破位：昨天或前天，至少有一天是收在 5 日线上的！
        cond_A_trend = (
            (df_base['close'] > df_base['ma5']) & 
            (df_base['ma5'] >= df_base['ma5_t1']) & 
            ((df_base['close_t1'] > df_base['ma5_t1']) | (df_base['close_t2'] > df_base['ma5_t2']))
        )

        df_type_A = df_base[cond_A_ignition & cond_A_volume & cond_A_money & cond_A_trend].copy()
        
        # 排序：优先主力资金，其次看总成交额，取前 3
        top_A = df_type_A.sort_values(by=['mf_3d_sum', 'amount'], ascending=[False, False]).head(3)
        codes_A = top_A['ts_code'].tolist()
        names_A = top_A['name'].tolist()

        # ==========================================
        # 策略二：【涨停反包龙头】(抽取 2 只)
        # ==========================================
        # 1. 近期有过涨停
        cond_B_limit = df_base['ts_code'].isin(limit_up_codes)
        
        # 2. 近一周上过龙虎榜 (增加防爆破：如果龙虎榜接口全没数据，则取消硬性要求)
        cond_B_top_list = df_base['ts_code'].isin(top_codes_5d) if top_codes_5d else True
        
        # 3. 调整后今日重新站稳 5日线：昨天或今天盘中破了5日线，但今天收盘站回来了！
        cond_B_trend = (df_base['close'] > df_base['ma5']) & ((df_base['close_t1'] <= df_base['ma5_t1']) | (df_base['low'] <= df_base['ma5'] * 1.02))
        
        # 4. 去重：已被策略A选中的不再选
        cond_B_distinct = ~df_base['ts_code'].isin(codes_A)

        df_type_B = df_base[cond_B_limit & cond_B_top_list & cond_B_trend & cond_B_distinct].copy()
        
        # 排序：妖股反包看绝对人气，按成交额降序取前 2
        top_B = df_type_B.sort_values(by='amount', ascending=False).head(2)
        codes_B = top_B['ts_code'].tolist()
        names_B = top_B['name'].tolist()

        # ==========================================
        # 终极合并输出
        # ==========================================
        final_codes = [c.split('.')[0] for c in (codes_A + codes_B)]
        final_names = names_A + names_B
        
        logging.info(f"📊 策略A(趋势核心)捕获: {list(zip([c.split('.')[0] for c in codes_A], names_A))}")
        logging.info(f"📊 策略B(妖股反包)捕获: {list(zip([c.split('.')[0] for c in codes_B], names_B))}")
        logging.info(f"🏆 最终合并自选标的池: {list(zip(final_codes, final_names))}")
        
        return final_codes

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
