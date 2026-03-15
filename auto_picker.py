import tushare as ts
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Tushare Pro
token = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(token)

def get_potential_stocks():
    """
    量化策略：【机构共振+主升浪启动模型】
    逻辑：主力资金大幅流入 + 龙虎榜机构席位买入 + 多头形态
    """
    # 自动获取最近一个交易日
    try:
        cal = pro.trade_cal(exchange='', is_open='1', start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d'), end_date=datetime.now().strftime('%Y%m%d'))
        last_trade_date = cal.iloc[-1]['cal_date']
    except:
        last_trade_date = datetime.now().strftime('%Y%m%d')

    try:
        logging.info(f"正在分析 {last_trade_date} 的主力机构动向...")
        
        # 1. 获取资金流向：筛选主力净流入 > 3000万 的个股
        df_flow = pro.moneyflow(trade_date=last_trade_date)
        df_flow = df_flow[df_flow['net_mf_amount'] > 3000] # 门槛：3000万起步

        # 2. 获取龙虎榜数据：寻找有机构席位买入的标的 (加分项)
        try:
            df_top = pro.top_list(trade_date=last_trade_date)
            # 筛选出机构买入额 > 0 的股票
            inst_stocks = df_top[df_top['inst_buy'] > 0]['ts_code'].unique().tolist()
        except:
            inst_stocks = []
            logging.warning("今日龙虎榜尚未更新，跳过机构席位硬过滤。")

        # 3. 获取行情数据进行形态过滤
        df_daily = pro.daily(trade_date=last_trade_date)
        df = pd.merge(df_flow, df_daily, on='ts_code')
        
        # 4. 组合策略筛选
        # - 涨幅在 3%~9% 之间（刚启动，未封死或刚突破）
        # - 收盘价 > 开盘价（收阳线，拒绝高开低走）
        condition = (df['pct_chg'] >= 3.0) & (df['pct_chg'] <= 9.0) & (df['close'] > df['open'])
        screened_df = df[condition].copy()

        # 5. 关联基础信息并过滤 ST
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,market')
        screened_df = pd.merge(screened_df, df_basic, on='ts_code')
        screened_df = screened_df[~screened_df['name'].str.contains('ST|退')]
        
        # 6. 计算最终得分：有机构席位买入的优先级最高，其次按主力净流入占比排序
        screened_df['is_inst'] = screened_df['ts_code'].apply(lambda x: 1 if x in inst_stocks else 0)
        # 排序逻辑：机构入驻优先 > 主力净流入额
        top_stocks = screened_df.sort_values(by=['is_inst', 'net_mf_amount'], ascending=False).head(5)
        
        stock_list = [code.split('.')[0] for code in top_stocks['ts_code'].tolist()]
        names = top_stocks['name'].tolist()
        
        logging.info(f"🚀 机构掘金完成！选出潜力标的: {list(zip(stock_list, names))}")
        return stock_list

    except Exception as e:
        logging.error(f"量化处理失败: {e}")
        return []

if __name__ == "__main__":
    if not token:
        logging.error("未检测到 TUSHARE_TOKEN")
    else:
        stocks = get_potential_stocks()
        if stocks:
            print(",".join(stocks))
