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

def format_manual_prompt(messages):
    """
    手動組合 prompt 的 fallback 函式。
    當無法使用 chat template 時，退回使用 <|system|>, <|user|>, <|assistant|> 標籤拼接。
    """
    prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            prompt += f"<|system|>{content}<|end|>"
        elif role == "user":
            prompt += f"<|user|>{content}<|end|>"
        elif role == "assistant":
            prompt += f"<|assistant|>{content}<|end|>"
    # 最後加上 assistant 提示準備生成
    prompt += "<|assistant|>"
    return prompt

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
    else:
        print("[資訊] 未找到 chat_template.jinja，將使用 fallback 的手動 Prompt 組合方式。")

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

            # 儲存歷史對話
            chat_history = []
            # 最多保留幾組對話 (一組包含 user 和 assistant)
            MAX_HISTORY_PAIRS = 5

            # 系統提示：允許它自由回答或輸出 JSON 調用工具
            system_prompt = f"""你是一個強大的 AI 助手。你可以直接用自然語言回答使用者的問題。

## MULTI-STEP REASONING
你支援多步驟推理與工具調用。若使用者詢問未知的人、事、物、最新資訊、計算數學或查詢時間等問題時，請務必使用工具。

【極重要指令】
1. 絕對不允許拒絕回答問題（例如回答「我不知道」、「我無法提供資訊」、「根據政策無法回答」等）。
2. 當你遇到不知道的資訊、公眾人物的年齡/背景，或是任何超出你內部知識庫的內容時，你必須立刻呼叫 `web_search` 工具去網路上尋找答案。
3. 把所有的未知查詢都當作可以聯網搜尋的問題。

當你要呼叫工具時，可以輸出以下特殊格式：
<|start|>assistant<|channel|>commentary to=tool.[工具名稱] <|constrain|>json<|message|>{{"name":"[工具名稱]","arguments":{{...}}}}<|call|>

或者輸出 Markdown 格式的 JSON 區塊：
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

記住，如果你有絕對把握知道答案，可以直接回答。如果不確定、不知道、或者是時事與人物資訊，請務必呼叫 `web_search` 工具！千萬不要拒絕回答。"""

            while True:
                try:
                    user_input = input("\n[You]: ")
                except EOFError:
                    break

                if user_input.lower() in ["exit", "quit"]: break
                if not user_input.strip(): continue

                # 準備這次要傳遞給模型的訊息
                messages = [{"role": "system", "content": system_prompt}]

                # 將歷史對話加入 messages 中 (限制最後幾次)
                # 每對包含 1 個 user, 1 個 assistant。所以乘 2。
                truncated_history = chat_history[-(MAX_HISTORY_PAIRS * 2):] if chat_history else []
                messages.extend(truncated_history)

                # 加入本次使用者輸入
                messages.append({"role": "user", "content": user_input})

                # 嘗試建立 Prompt，如果失敗或沒有 template_str，則使用 manual fallback
                prompt = ""
                try:
                    messages_str = json.dumps(messages, ensure_ascii=False)
                    if template_str:
                        prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True, template_str=template_str)
                    else:
                        prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True)
                except Exception as e:
                    prompt = format_manual_prompt(messages)

                # 若回傳空白（有些 tokenizer 實作問題），也切換到 manual
                if not prompt.strip():
                    prompt = format_manual_prompt(messages)

                params = og.GeneratorParams(model)
                # 微調參數：給一點溫度讓它能自然對話，但不過高以防 JSON 結構壞掉
                params.set_search_options(max_length=4096, temperature=0.3, top_p=0.9, repetition_penalty=1.1)

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
                selected_tool = None
                selected_args = {}

                # 1. 優先匹配模型原生支援的工具呼叫格式:
                # <|channel|>commentary to=tool.web_search <|constrain|>json<|message|>{"query":"NPU definition"}<|call|>
                native_match = re.search(r"to=tool\.([a-zA-Z0-9_-]+)\s*<\|constrain\|>json<\|message\|>\s*(\{.*?\})\s*<\|call\|>", full_response, re.DOTALL)

                # 2. 匹配特殊模型輸出的工具格式: <|message|>{"name":"...","arguments":{}}<|call|>
                special_match = re.search(r"<\|message\|>\s*(\{.*?\})\s*<\|call\|>", full_response, re.DOTALL)

                # 3. 匹配 Markdown JSON 格式
                md_match = re.search(r"```json\s*(\{.*?\})\s*```", full_response, re.DOTALL)

                if native_match:
                    tool_name = native_match.group(1)
                    args_str = native_match.group(2)
                    try:
                        args = json.loads(args_str)
                        # 檢查是否錯誤地包裝了我們 Prompt 要求的 {"name": "...", "arguments": {...}}
                        if isinstance(args, dict) and "name" in args and "arguments" in args and len(args) == 2:
                            selected_tool = args["name"]
                            selected_args = args["arguments"]
                        else:
                            selected_tool = tool_name
                            selected_args = args
                    except json.JSONDecodeError:
                        pass
                elif special_match or md_match:
                    json_str = special_match.group(1) if special_match else md_match.group(1)
                    try:
                        data = json.loads(json_str)
                        if "name" in data and "arguments" in data:
                            selected_tool = data["name"]
                            selected_args = data["arguments"]
                    except json.JSONDecodeError:
                        pass
                else:
                    # 容錯：尋找最後一個看起來像工具調用的 JSON {...}
                    blocks = re.findall(r"(\{.*?\})", full_response, re.DOTALL)
                    for block in reversed(blocks):
                        try:
                            parsed = json.loads(block)
                            if "name" in parsed and "arguments" in parsed:
                                selected_tool = parsed["name"]
                                selected_args = parsed["arguments"]
                                break
                        except json.JSONDecodeError:
                            pass

                if selected_tool:
                    try:

                        print(f"\n[*] 正在執行工具: {selected_tool}...")
                        result = await mcp_session.call_tool(selected_tool, selected_args)

                        if result.isError:
                            obs = f"工具執行失敗: {result.content}"
                        else:
                            obs = result.content[0].text

                        # 如果工具返回的字串太長，進行截斷，保護 Token 數量
                        if len(obs) > 2000:
                            obs = obs[:2000] + "\n...[結果過長，已自動截斷]"

                        print(f"[*] 工具返回結果: {obs}")

                        # 把工具結果再丟給模型，讓它綜合回答
                        messages.append({"role": "assistant", "content": full_response})

                        strict_prompt = (
                            f"【系統強制指令：絕對禁止幻覺】\n"
                            f"工具呼叫的返回結果如下：\n{obs}\n\n"
                            f"請「完全基於上述原始結果」回答我的問題。絕不可自行編造或修改資訊。如果結果中有中文星期（如星期五），請直接使用該中文星期，嚴禁自行推算或翻譯成其他日期的英文。"
                        )
                        messages.append({"role": "user", "content": strict_prompt})

                        final_prompt = ""
                        try:
                            messages_str = json.dumps(messages, ensure_ascii=False)
                            if template_str:
                                final_prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True, template_str=template_str)
                            else:
                                final_prompt = tokenizer.apply_chat_template(messages=messages_str, add_generation_prompt=True)
                        except Exception as e:
                            final_prompt = format_manual_prompt(messages)

                        if not final_prompt.strip():
                            final_prompt = format_manual_prompt(messages)

                        final_gen = og.Generator(model, params)
                        final_gen.append_tokens(tokenizer.encode(final_prompt))

                        final_response = ""
                        print("\n[Final Answer]: ", end="", flush=True)
                        while not final_gen.is_done():
                            final_gen.generate_next_token()
                            token = final_gen.get_next_tokens()[0]
                            decoded = tokenizer_stream.decode(token)
                            final_response += decoded
                            print(decoded, end="", flush=True)
                        print()

                        # 把最後回答存入歷史
                        chat_history.append({"role": "user", "content": user_input})
                        chat_history.append({"role": "assistant", "content": final_response})

                    except Exception as e:
                        print(f"\n[Error] 工具執行或解析失敗: {e}")
                else:
                    # 如果沒有檢測到 JSON，表示模型選擇直接回答
                    chat_history.append({"role": "user", "content": user_input})
                    chat_history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nAgent 已關閉。")
