import onnxruntime_genai as og
import asyncio, re, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 確保輸出編碼正確，防止 Windows 崩潰
sys.stdout.reconfigure(encoding='utf-8')

# --- 配置區 ---
MODEL_PATH = r"C:\Users\test\Desktop\LLM_Test\gpt-oss-20b-onnx-ryzenai-npu"
TOOL_SCRIPT_PATH = r"C:\Users\test\Desktop\LLM_Test\python_script\mytool.py"

async def run_agent():
    print(f"[*] 正在初始化 NPU 推理引擎與 MCP 工具伺服器...")
    model = og.Model(MODEL_PATH)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    server_params = StdioServerParameters(command="python", args=[TOOL_SCRIPT_PATH])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            print("\n" + "="*50 + "\n🚀 深度解析版 Agent 已上線 (支援數學/時間/天氣自動校正)\n" + "="*50)

            while True:
                user_input = input("\n[You]: ")
                if user_input.lower() in ["exit", "quit"]: break

                # 強化指令：告訴模型它有 2026 年權限，並要求結構化輸出
                prompt = f"<|system|>你是 2026 年的 AI 助手。遇到數學、時間、天氣問題，請「僅輸出」JSON 調用格式。\n格式：{{\"name\": \"工具名\", \"arguments\": {{\"參數\": \"值\"}}}}<|end|><|user|>{user_input}<|end|><|assistant|>"

                params = og.GeneratorParams(model)
                # 關鍵：溫度設為 0 以保證數學與邏輯穩定，repetition_penalty 防止重複輸出
                params.set_search_options(max_length=512, temperature=0.0, repetition_penalty=1.2)
                
                generator = og.Generator(model, params)
                generator.append_tokens(tokenizer.encode(prompt))

                full_response = ""
                print("[Thinking...]", end=" ", flush=True)

                # 第一階段：攔截生成
                while not generator.is_done():
                    generator.generate_next_token()
                    token = generator.get_next_tokens()[0]
                    decoded = tokenizer_stream.decode(token)
                    full_response += decoded
                    print(decoded, end="", flush=True)
                    
                    # 只要看到工具啟動的徵兆（JSON、ACTION 或 call），立刻停下來抓取後續內容
                    if any(trigger in full_response for trigger in ["{", "ACTION:", "call"]):
                        # 額外多抓一些 token 確保 JSON 內容被完整讀取
                        for _ in range(30):
                            if generator.is_done(): break
                            generator.generate_next_token()
                            full_response += tokenizer_stream.decode(generator.get_next_tokens()[0])
                        break

                # 第二階段：深度解析 (從雜訊中挖取 JSON)
                selected_tool = None
                selected_args = {}

                # 嘗試從全體回覆中尋找最後一個 JSON 區塊
                json_blocks = re.findall(r"(\{.*?\})", full_response, re.DOTALL)
                if json_blocks:
                    try:
                        # 嘗試解析最後一個 JSON
                        data = json.loads(json_blocks[-1])
                        # 適應模型自發產生的各種標籤 (name/action/arguments/query)
                        selected_tool = data.get("name") or data.get("action") or data.get("tool")
                        selected_args = data.get("arguments") or data.get("args") or data
                        
                        # 針對數學運算的自動判定 (如果模型只噴了 expression)
                        if not selected_tool and "expression" in data:
                            selected_tool = "calculate_math"
                    except: pass
                
                # 若無 JSON 則嘗試正則抓取 ACTION
                if not selected_tool:
                    action_match = re.search(r"ACTION:\s*(\w+)", full_response)
                    if action_match: selected_tool = action_match.group(1)

                # 第三階段：執行與強制回覆
                if selected_tool:
                    print(f"\n[*] 執行工具: {selected_tool}...")
                    try:
                        result = await mcp_session.call_tool(selected_tool, selected_args)
                        obs = result.content[0].text
                        
                        # 強制 Pre-filling 技術：給模型一個無法拒絕的開頭
                        final_prompt = f"<|system|>你現在是 2026 年的助手。請直接根據數據回答，不准說不知道。數據：{obs}<|end|><|user|>{user_input}<|end|><|assistant|>答案是："
                        
                        final_gen = og.Generator(model, params)
                        final_gen.append_tokens(tokenizer.encode(final_prompt))
                        print("\n[Final Answer]: 答案是：", end="")
                        while not final_gen.is_done():
                            final_gen.generate_next_token()
                            print(tokenizer_stream.decode(final_gen.get_next_tokens()[0]), end="", flush=True)
                        print()
                    except Exception as e:
                        print(f"\n[Error]: 工具執行或解析失敗: {e}")
                else:
                    # 如果模型沒打算用工具，直接輸出結果
                    if "Assistant" in full_response:
                        print(f"\n[Direct]: {full_response.split('Assistant')[-1]}")
                    else:
                        print(f"\n[Direct]: {full_response}")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nAgent 已關閉。")