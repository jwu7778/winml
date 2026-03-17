# Local NPU AI Agent (AMD Ryzen AI + MCP Tools)

這是一個基於 **AMD Ryzen AI NPU** 和 **ONNXRuntime GenAI (OGA)** 所打造的地端（完全離線運行）大語言模型代理人 (Local AI Agent) 專案。

本專案將地端的 LLM 當作「大腦 (`brain.py`)」，負責自然語言理解與多步驟推理，並透過 **Model Context Protocol (MCP)** 架構連接外部工具 (`mytool.py`)，讓模型能夠自主判斷何時需要聯網搜尋、計算數學、或查詢精確的世界時間。

## 🌟 核心特色

1. **完全地端與隱私安全**：語言模型 (如 `gpt-oss-20b-onnx-ryzenai-npu`) 完全在本地端 AMD NPU 執行，資料不需上傳雲端，保障企業與個人隱私。
2. **多步驟推理與工具調用 (Multi-Step Reasoning)**：模型被賦予了自主思考能力。當遇到未知的資訊、最新時事或需要運算時，它會主動輸出特定格式來呼叫工具，獲取結果後再給出最終答案。
3. **零幻覺機制**：系統層面實作了嚴格的強制指令與提示詞工程 (Prompt Engineering)，確保模型「完全基於工具返回的真實結果」回答問題，禁止自行編造或推算（如日期、星期）。
4. **資源最佳化**：
   * 動態管理對話歷史 (Rolling Window)，避免 Token 無限膨脹。
   * 工具返回結果自動截斷機制 (防止 `web_search` 抓取過長網頁導致 OOM)。
   * 針對 32GB 記憶體環境與 13GB 模型的最佳化 `max_length` 設定。

## 🛠️ 內建工具 (MCP Tools)

本專案透過 `mcp.server.fastmcp` 提供以下強大的工具給模型使用：

*   **`web_search` (即時聯網搜尋)**
    *   **用途**：當使用者詢問最新新聞、未知事實、公眾人物資訊時，Agent 會自動呼叫此工具。
    *   **實作**：底層使用 `duckduckgo_search` (DDGS) 進行無痕搜尋，並強化關鍵字過濾以獲取最相關的結果。
*   **`get_current_time` (精準時間感知)**
    *   **用途**：查詢現在的時間。支援「本機時間」與「指定國家/城市時間」。
    *   **實作**：
        *   無參數時：回傳本機系統時間。
        *   帶入地點參數 (例如：台灣、東京、New York) 時：自動透過 `timeapi.io` 取得該地區 100% 絕對準確的真實時間與星期，徹底解決模型靠搜尋引擎查時間容易抓到舊新聞的問題。
*   **`calculate_math` (精準數學計算)**
    *   **用途**：解決 LLM 容易算錯數學的通病。當需要進行數學運算（如 `3*5`, `sqrt(16)`）時，模型會將算式交給此工具。
    *   **實作**：使用 Python 內建的安全 `eval` 結合 `math` 函式庫計算出絕對正確的結果。

## 🚀 環境與安裝需求

### 系統與硬體
*   **OS**: Windows 11 (建議)
*   **硬體**: 搭載 AMD Ryzen AI NPU 的處理器 (例如 AMD Ryzen 8000/AI 300 系列)
*   **記憶體**: 建議 32GB 以上

### 軟體依賴
請確保您已安裝最新版的 Python (推薦 3.10+)，並安裝以下套件：

```bash
pip install onnxruntime-genai
pip install mcp
pip install duckduckgo-search
```

### 模型準備
請從 Hugging Face 下載相容 AMD NPU 的 ONNX 模型（例如 `amd/gpt-oss-20b-onnx-ryzenai-npu`），並將模型資料夾路徑記下。

## 🎮 如何執行

此專案分為「工具伺服器」與「大腦代理人」兩個部分。您需要準備兩個終端機視窗。

### 1. 啟動 MCP 工具伺服器
在第一個終端機中，執行 `mytool.py`。這將會啟動 MCP 伺服器並透過標準輸入輸出等待指令。

*(註：由於 `brain.py` 目前的設計是透過子處理程序 (Subprocess) 自動帶起工具，您可能只需要執行 `brain.py` 即可。視您對 MCP Client 的實作方式而定。若您使用 STDIO 傳輸，`brain.py` 內部已設定 `command="python", args=["mytool.py"]` 會自動啟動。)*

### 2. 啟動 AI Agent 大腦
在終端機中，設定環境變數以指定模型路徑（或者您可以直接修改 `brain.py` 內的預設路徑）。

**Windows (PowerShell):**
```powershell
$env:MODEL_DIR="C:\path\to\your\onnx\model\folder"
python brain.py
```

啟動後，您會看到模型載入進度，以及工具掛載成功的提示：

```text
==================================================
🚀 深度解析版 Agent 已上線 (支援動態工具呼叫)
[*] 載入的工具數量: 3
==================================================

[You]: 現在台灣時間是幾點？
```

接著，您就可以開始用自然語言與您的地端 NPU AI 助理對話了！它會自動判斷何時該呼叫工具來回答您的問題。

## ⚙️ 進階配置與故障排除

*   **終端機編碼問題 (Windows)**：若您在 Windows 上遇到中文輸出變成亂碼或報錯，專案內的程式碼已加入 `sys.stdout.reconfigure(encoding='utf-8')` 自動修復。
*   **記憶體不足 (OOM)**：如果對話到一半崩潰，請檢查 `brain.py` 中的 `max_length=4096` 設定。如果您的記憶體較小，可以嘗試調降此數值，但可能會截斷較長的思考過程。
*   **API 網路錯誤**：查詢國外時間時，若遇到網路錯誤，請檢查防火牆是否阻擋了 `timeapi.io` 的連線。

---
*Powered by AMD Ryzen™ AI & Model Context Protocol*
