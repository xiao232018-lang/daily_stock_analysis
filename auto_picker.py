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
    量化策略：【主升浪·主力箱体吸筹与点火模型】(终极稳定版)
    核心逻辑：
    1. 资金面：近 5 天内至少 3 天主力净流入且总额 > 5000万。
    2. 形态面：近 15 天内股价振幅 < 25%，证明主力在控盘压价吸筹。
    3. 启动面：今日温和洗盘或点火突破（涨跌幅 -3% ~ 9%）。
    4. 基础面：主板、非ST、股价<200、换手率 3%~20%。
    """
    try:
        # 1. 划定时间窗口：取近 15 个交易日和近 5 个交易日
        cal = pro.trade_cal(exchange='', is_open='1', 
                            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), 
                            end_date=datetime.now().strftime('%Y%m%d'))
        dates_15 = cal['cal_date'].tolist()[-15:]
        dates_5 = dates_15[-5:]
        last_date = dates_15[-1]
        
        logging.info(f"正在扫描全市场，分析时间窗口: {dates_15[0]} 至 {last_date}")

        # ==========================================
        # 模块一：【资金连续潜伏探测】
        # ==========================================
        df_mf_list = [pro.moneyflow(trade_date=d) for d in dates_5]
        df_mf_5d = pd.concat(df_mf_list)
        
        # 标记每天是否有主力净流入 (>0)
        df_mf_5d['is_inflow'] = df_mf_5d['net_mf_amount'].apply(lambda x: 1 if x > 0 else 0)
        
        # 聚合计算近 5 天的总流入额，以及“正向流入的天数”
        mf_agg = df_mf_5d.groupby('ts_code').agg(
            mf_5d_sum=('net_mf_amount', 'sum'),
            inflow_days=('is_inflow', 'sum')
        ).reset_index()
        
        # 核心条件：5天内至少3天净买入，且累计净流入 > 5000 万
        cond_money = (mf_agg['inflow_days'] >= 3) & (mf_agg['mf_5d_sum'] > 5000)
        valid_mf_stocks = mf_agg[cond_money]

        if valid_mf_stocks.empty:
            logging.warning("未检测到明显的连续吸筹资金，提前结束筛选。")
            return []

        # ==========================================
        # 模块二：【箱体压盘探测】
        # ==========================================
        target_codes = valid_mf_stocks['ts_code'].tolist()
        
        df_daily_list = [pro.daily(trade_date=d) for d in dates_15]
        df_daily_15d = pd.concat(df_daily_list)
        # 仅保留有资金潜伏的股票，加快计算
        df_daily_15d = df_daily_15d[df_daily_15d['ts_code'].isin(target_codes)]
        
        # 计算 15 天内的最高价、最低价
        price_agg = df_daily_15d.groupby('ts_code').agg(
            max_high=('high', 'max'),
            min_low=('low', 'min')
        ).reset_index()
        
        # 振幅 = (区间最高 - 区间最低) / 区间最低 * 100
        price_agg['amplitude_15d'] = (price_agg['max_high'] - price_agg['min_low']) / price_agg['min_low'] * 100
        
        # 核心条件：15天内上下振幅 <= 25% (主力控盘压价)
        cond_box = price_agg['amplitude_15d'] <= 25.0
        valid_box_stocks = price_agg[cond_box]

        # 增加防崩溃拦截：如果没票符合箱体要求，直接退出
        if valid_box_stocks.empty:
            logging.warning("资金潜伏标的中，未检测到符合【箱体压盘】特征的股票。")
            return []

        # ==========================================
        # 模块三：【基础过滤与龙虎榜加分】
        # ==========================================
        # 获取最新一天的行情
        df_today = pro.daily(trade_date=last_date)
        # 获取基本面 (🚨 注意：这里已经去掉了 close，完美解决了 KeyError 问题)
        df_basic = pro.daily_basic(trade_date=last_date, fields='ts_code,turnover_rate,total_mv')
        df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market')
        
        # 多表合并
        df = pd.merge(valid_mf_stocks, valid_box_stocks, on='ts_code')
        df = pd.merge(df, df_today, on='ts_code')
        df = pd.merge(df, df_basic, on='ts_code')
        df = pd.merge(df, df_info, on='ts_code')

        # 龙虎榜机构数据探测 (近3日有无机构真金白银上榜)
        inst_stocks = []
        for d in dates_15[-3:]:
            try:
                top = pro.top_list(trade_date=d)
                inst_stocks.extend(top[top['inst_buy'] > 0]['ts_code'].tolist())
            except: 
                pass

        # ==========================================
        # 执行最终精准过滤
        # ==========================================
        # 1. 主板非 ST 且 价格 < 200
        cond_board = df['ts_code'].str.match(r'^(60|00)')
        cond_not_st = ~df['name'].str.contains('ST|退')
        cond_price = df['close'] < 200
        
        # 2. 换手率 3% ~ 20% (建仓/初升健康换手)
        cond_turnover = (df['turnover_rate'] >= 3.0) & (df['turnover_rate'] <= 20.0)
        
        # 3. K线洗盘与点火突破 (-3% 到 9%)
        cond_kline = (df['pct_chg'] >= -3.0) & (df['pct_chg'] <= 9.0)

        # 综合应用过滤条件
        screened_df = df[cond_board & cond_not_st & cond_price & cond_turnover & cond_kline].copy()

        if screened_df.empty:
            logging.warning("经过最终基本面与形态过滤，今日无符合条件的标的。")
            return []

        # 计算打分排序：机构进场优先 > 5日潜伏资金总额
        screened_df['is_inst'] = screened_df['ts_code'].apply(lambda x: 1 if x in inst_stocks else 0)
        top_stocks = screened_df.sort_values(by=['is_inst', 'mf_5d_sum'], ascending=False).head(5)

        # 清洗代码格式 (例如把 000001.SZ 变成 000001 发给主程序)
        stock_list = [code.split('.')[0] for code in top_stocks['ts_code'].tolist()]
        names = top_stocks['name'].tolist()
        
        logging.info(f"🎯 主力吸筹模型扫描完毕！精选标的: {list(zip(stock_list, names))}")
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
            # 这一行是给 GitHub Actions 捕获用的，绝对不要加多余的 print
            print(",".join(stocks))
