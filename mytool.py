import sys
import json
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timezone, timedelta
try:
    from duckduckgo_search import DDGS
except ImportError:
    # 某些環境下可能需要直接安裝 ddgs 並這樣匯入
    from ddgs import DDGS

# --- 核心修正：解決 Windows 終端機編碼與輸出緩衝問題 ---
sys.stdout.reconfigure(encoding='utf-8')

# 創建 MCP 伺服器實例
mcp = FastMCP("AdvancedAgentBrain")

# --- 工具 1：精準數學計算 ---
@mcp.tool()
def calculate_math(expression: str) -> str:
    """
    當需要執行數學運算（如 3*5, sqrt(16)）時調用。
    """
    import math
    try:
        # 使用 eval 處理基礎運算，加入 math 函式庫支援
        result = eval(expression, {"__builtins__": None}, {"sqrt": math.sqrt, "pow": pow})
        return f"計算結果為: {result}"
    except Exception as e:
        return f"計算出錯: {str(e)}"

# --- 工具 2：即時時間感知 ---
@mcp.tool()
def get_current_time() -> str:
    """
    當用戶詢問現在的時間、日期或今天是星期幾時調用。
    """
    tz_tw = timezone(timedelta(hours=8))
    now = datetime.now(tz_tw)
    weekday_map = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}
    return now.strftime(f"%Y-%m-%d %H:%M:%S 星期{weekday_map[now.weekday()]}")

# --- 工具 3：真實聯網搜尋 (強化過濾與區域鎖定版) ---
@mcp.tool()
def web_search(query: str) -> str:
    """當用戶詢問即時新聞、天氣、或資訊時調用。"""
    try:
        clean_query = query.strip()
        # 修正 2: 強化 query，加入明確的標籤引導搜尋引擎
        search_intent = f"{clean_query} Taiwan news"
        
        print(f"[*] 執行搜尋: {search_intent}")
        
        # 修正 3: 調整搜尋參數，移除過於嚴格的 timelimit 以避免回傳雜訊
        with DDGS() as ddgs:
            search_results = ddgs.text(
                search_intent, 
                region='tw-tzh',  
                safesearch='off', 
                # 如果找不到結果，不要限制在 'd'，改用 None 讓 DDG 給出最相關的
                timelimit=None,    
                max_results=3
            )
            
            results = []
            if search_results:
                for r in search_results:
                    results.append(f"標題: {r['title']}\n內容: {r['body']}")
            
            if not results:
                return f"找不到關於 '{clean_query}' 的即時資訊。建議您稍後再試或更換關鍵字。"
            
            return "\n\n".join(results)
            
    except Exception as e:
        return f"搜尋錯誤: {str(e)}"

if __name__ == "__main__":
    mcp.run()