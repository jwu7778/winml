import onnxruntime_genai as og
import asyncio, re, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 確保輸出編碼正確，防止 Windows 崩潰
sys.stdout.reconfigure(encoding='utf-8')

# --- 配置區 ---
# 支援透過環境變數覆寫，若無則使用預設路徑
MODEL_PATH = os.environ.get("MODEL_PATH", r"C:\Users\test\Desktop\LLM_Test\gpt-oss-20b-onnx-ryzenai-npu")
# 預設使用同目錄下的 mytool.py
DEFAULT_TOOL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mytool.py")
TOOL_SCRIPT_PATH = os.environ.get("TOOL_SCRIPT_PATH", DEFAULT_TOOL_PATH)

async def run_agent():
    print(f"[*] 正在初始化 NPU 推理引擎與 MCP 工具伺服器...")
    print(f"[*] 模型路徑: {MODEL_PATH}")
    print(f"[*] 工具腳本: {TOOL_SCRIPT_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"[警告] 模型路徑 {MODEL_PATH} 不存在，請確認路徑設定。")
    if not os.path.exists(TOOL_SCRIPT_PATH):
        print(f"[警告] 工具腳本 {TOOL_SCRIPT_PATH} 不存在，請確認路徑設定。")

    try:
        model = og.Model(MODEL_PATH)
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
    except Exception as e:
        print(f"[錯誤] 載入模型失敗: {e}")
        return

    # 嘗試讀取 chat_template.jinja
    jinja_path = os.path.join(MODEL_PATH, "chat_template.jinja")
    template_str = None
    if os.path.exists(jinja_path):
        with open(jinja_path, "r", encoding="utf-8") as f:
            template_str = f.read()

    server_params = StdioServerParameters(command="python", args=[TOOL_SCRIPT_PATH])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()

            # 動態獲取工具清單
            tools_response = await mcp_session.list_tools()
            tools_list = []
            if tools_response and hasattr(tools_response, 'tools'):
                for tool in tools_response.tools:
                    tools_list.append({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    })
            tools_json = json.dumps(tools_list, ensure_ascii=False, indent=2)

            print("\n" + "="*50)
            print("🚀 深度解析版 Agent 已上線 (支援動態工具呼叫)")
            print(f"[*] 載入的工具數量: {len(tools_list)}")
            print("="*50 + "\n")

            while True:
                try:
                    user_input = input("\n[You]: ")
                except EOFError:
                    break

                if user_input.lower() in ["exit", "quit"]: break
                if not user_input.strip(): continue

                # 系統提示：允許它自由回答或輸出 JSON 調用工具
                system_prompt = f"""你是一個強大的 AI 助手。你可以直接用自然語言回答使用者的問題。
如果你需要搜尋最新資訊、計算數學或查詢時間，你可以使用工具。
當你要使用工具時，請「僅」輸出以下格式的 JSON 區塊（請包含 ```json 和 ``` 標籤）：
```json
{{
  "name": "工具名稱",
  "arguments": {{
    "參數1": "值1"
  }}
}}
```

可用的工具如下：
{tools_json}

記住，如果你知道答案，可以直接回答。如果需要外部資訊或計算，再呼叫對應的工具。"""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
                messages_str = json.dumps(messages, ensure_ascii=False)

                if template_str:
                    prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True, template_str=template_str)
                else:
                    prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True)

                params = og.GeneratorParams(model)
                # 微調參數：給一點溫度讓它能自然對話，但不過高以防 JSON 結構壞掉
                params.set_search_options(max_length=1024, temperature=0.3, top_p=0.9, repetition_penalty=1.1)

                generator = og.Generator(model, params)
                generator.append_tokens(tokenizer.encode(prompt))

                full_response = ""
                print("[Agent]: ", end="", flush=True)

                while not generator.is_done():
                    generator.generate_next_token()
                    token = generator.get_next_tokens()[0]
                    decoded = tokenizer_stream.decode(token)
                    full_response += decoded
                    print(decoded, end="", flush=True)
                print()

                # 尋找 JSON 工具呼叫
                tool_call_json = None
                match = re.search(r"```json\s*(\{.*?\})\s*```", full_response, re.DOTALL)
                if match:
                    tool_call_json = match.group(1)
                else:
                    # 容錯：尋找最後一個看起來像工具調用的 JSON {...}
                    blocks = re.findall(r"(\{.*?\})", full_response, re.DOTALL)
                    for block in reversed(blocks):
                        try:
                            parsed = json.loads(block)
                            if "name" in parsed and "arguments" in parsed:
                                tool_call_json = block
                                break
                        except json.JSONDecodeError:
                            pass

                if tool_call_json:
                    try:
                        data = json.loads(tool_call_json)
                        selected_tool = data.get("name")
                        selected_args = data.get("arguments", {})

                        print(f"\n[*] 正在執行工具: {selected_tool}...")
                        result = await mcp_session.call_tool(selected_tool, selected_args)

                        if result.isError:
                            obs = f"工具執行失敗: {result.content}"
                        else:
                            obs = result.content[0].text

                        print(f"[*] 工具返回結果: {obs}")

                        # 把工具結果再丟給模型，讓它綜合回答
                        messages.append({"role": "assistant", "content": full_response})
                        messages.append({"role": "user", "content": f"工具呼叫的返回結果如下：\n{obs}\n請根據上述資訊回答我的問題。"})

                        messages_str = json.dumps(messages, ensure_ascii=False)
                        if template_str:
                            final_prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True, template_str=template_str)
                        else:
                            final_prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True)

                        final_gen = og.Generator(model, params)
                        final_gen.append_tokens(tokenizer.encode(final_prompt))

                        print("\n[Final Answer]: ", end="", flush=True)
                        while not final_gen.is_done():
                            final_gen.generate_next_token()
                            print(tokenizer_stream.decode(final_gen.get_next_tokens()[0]), end="", flush=True)
                        print()

                    except Exception as e:
                        print(f"\n[Error] 工具執行或解析失敗: {e}")
                else:
                    # 如果沒有檢測到 JSON，表示模型選擇直接回答
                    pass

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nAgent 已關閉。")