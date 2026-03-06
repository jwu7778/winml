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
def get_current_time(location: str = None) -> str:
    """
    當用戶詢問「現在的時間、日期或星期」時調用。絕對不要用 web_search 來查時間。

    參數:
    - location (str, optional): 如果用戶詢問「特定國家或城市」的時間（例如"台灣"、"東京"、"紐約"、"London"），請傳入該地點名稱。如果只是問「現在幾點」，請留空 (None)。

    回傳:
    - 該地區或本機的精確時間與星期。
    """
    import urllib.request
    import urllib.parse

    weekday_map = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}

    # 若沒有指定地點，回傳系統預設時間 (這裡我們以當地系統時間為基準)
    if not location or location.strip() == "":
        now = datetime.now()
        local_time_str = now.strftime(f"%Y-%m-%d %H:%M:%S 星期{weekday_map[now.weekday()]}")
        return f"本機系統時間: {local_time_str}"

    # 若有指定地點，我們透過簡單的免費 API 或邏輯處理。
    # 這裡為求穩定，使用 http://worldtimeapi.org 或類似服務的替代邏輯。
    # 為了避免 API 不穩定或名稱配對困難，我們串接 TimeApi (https://timeapi.io/) 支援地區搜尋，
    # 或是最簡單的：將 location 轉換為常見 timezone 來查詢 WorldTimeAPI。
    # 因為 LLM 傳入的可能是中文（如 "台灣", "紐約"），我們做個簡單的 mapping，
    # 如果找不到，就 fallback 到本機時間加上提示。

    loc_lower = location.lower()

    # 簡易時區映射 (針對台灣與常見城市)
    timezone_mapping = {
        "taiwan": "Asia/Taipei",
        "taipei": "Asia/Taipei",
        "台灣": "Asia/Taipei",
        "臺北": "Asia/Taipei",
        "台北": "Asia/Taipei",
        "tokyo": "Asia/Tokyo",
        "東京": "Asia/Tokyo",
        "japan": "Asia/Tokyo",
        "日本": "Asia/Tokyo",
        "seoul": "Asia/Seoul",
        "首爾": "Asia/Seoul",
        "韓國": "Asia/Seoul",
        "new york": "America/New_York",
        "紐約": "America/New_York",
        "london": "Europe/London",
        "倫敦": "Europe/London",
        "paris": "Europe/Paris",
        "巴黎": "Europe/Paris",
        "sydney": "Australia/Sydney",
        "雪梨": "Australia/Sydney"
    }

    # 找出對應的時區字串
    tz_string = None
    for key, tz in timezone_mapping.items():
        if key in loc_lower:
            tz_string = tz
            break

    # 如果找不到 mapping，我們嘗試回報不支援，或是提供系統時間
    if not tz_string:
        return f"無法解析地點 '{location}' 的時區。建議使用 web_search 查詢該地區的當前時區偏移量，或是請提供更明確的城市名稱。"

    # 我們改用 timeapi.io 來處理時區問題，因為 worldtimeapi 在某些環境下回傳 410 Gone 或阻擋連線。
    # API endpoint: https://timeapi.io/api/Time/current/zone?timeZone=Asia/Taipei
    try:
        url = f"https://timeapi.io/api/Time/current/zone?timeZone={tz_string}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            # timeapi.io 回傳格式範例:
            # {"year":2024,"month":3,"day":7,"hour":15,"minute":30,"seconds":45,"dayOfWeek":"Thursday",...}

            year = data.get("year")
            month = data.get("month")
            day = data.get("day")
            hour = data.get("hour")
            minute = data.get("minute")
            second = data.get("seconds")
            day_of_week = data.get("dayOfWeek", "")

            # 將英文星期轉換為中文
            dow_mapping = {
                "Monday": "一",
                "Tuesday": "二",
                "Wednesday": "三",
                "Thursday": "四",
                "Friday": "五",
                "Saturday": "六",
                "Sunday": "日"
            }
            weekday_str = dow_mapping.get(day_of_week, day_of_week)

            return f"{location} 當地時間: {year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d} 星期{weekday_str}"

    except Exception as e:
        return f"查詢 {location} ({tz_string}) 的時間時發生網路或解析錯誤: {str(e)}。請確認網路連線或稍後再試。"

# --- 工具 3：真實聯網搜尋 (強化過濾與區域鎖定版) ---
@mcp.tool()
def web_search(query: str) -> str:
    """當用戶詢問即時新聞、天氣、或資訊時調用。"""
    try:
        clean_query = query.strip()
        # 修正 2: 強化 query，加入明確的標籤引導搜尋引擎
        search_intent = clean_query
        
        print(f"[*] 執行搜尋: {search_intent}")
        
        # 修正 3: 調整搜尋參數，移除過於嚴格的 timelimit 以避免回傳雜訊
        with DDGS() as ddgs:
            search_results = ddgs.text(
                search_intent, 
                region='wt-wt',
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