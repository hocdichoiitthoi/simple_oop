import os
import csv
import base64
import json
from typing import Literal, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv() 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class BusinessCard(BaseModel):
    name: Optional[str] = Field(description="Tên đầy đủ")
    company: Optional[str] = Field(description="Tên công ty")
    address: Optional[str] = Field(description="Địa chỉ")
    phone_number: Optional[str] = Field(description="Số điện thoại")
    email: Optional[str] = Field(description="Email")
    job_title: Optional[str] = Field(description="Chức danh")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
vision_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(BusinessCard)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    extracted_data: Optional[dict] # Lưu tạm dữ liệu sau khi OCR để Node Save dùng

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chatbot(state: State):
    print("--- NODE: CHATBOT ---")
    # Bind tool giả để LLM biết nó có khả năng OCR
    # Lưu ý: Chúng ta không dùng bind_tools để chạy tự động, mà để LLM biết "khi nào nên dùng"
    tool_schema = {
        "name": "ocr_extractor",
        "description": "Sử dụng khi người dùng cung cấp hình ảnh danh thiếp cần trích xuất thông tin.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Đường dẫn tới file ảnh"}
            },
            "required": ["image_path"]
        }
    }
    
    llm_with_tools = llm.bind_tools([tool_schema])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def ocr_processing(state: State):
    print("--- NODE: OCR PROCESSING ---")
    last_message = state["messages"][-1]
    
    # Lấy thông tin tool call từ LLM
    tool_call = last_message.tool_calls[0]
    image_path = tool_call["args"]["image_path"]
    
    try:
        # Gọi GPT Vision (Logic xử lý ảnh)
        base64_img = encode_image(image_path)
        msg = HumanMessage(content=[
            {"type": "text", "text": "Trích xuất thông tin danh thiếp này."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ])
        
        # Kết quả trả về dạng Object BusinessCard
        result = vision_llm.invoke([msg])
        data_dict = result.model_dump()
        
        # Tạo thông báo phản hồi (ToolMessage)
        tool_msg = ToolMessage(
            tool_call_id=tool_call["id"],
            content=f"Đã trích xuất thành công: {json.dumps(data_dict, ensure_ascii=False)}"
        )
        
        return {"messages": [tool_msg], "extracted_data": data_dict}
        
    except Exception as e:
        return {"messages": [ToolMessage(tool_call_id=tool_call["id"], content=f"Lỗi: {str(e)}")]}

def save_to_csv(state: State):
    print("--- NODE: SAVE CSV ---")
    data = state.get("extracted_data")
    
    if data:
        file_exists = os.path.isfile(Path(__file__).parent / 'data.csv')
        with open(Path(__file__).parent / 'data.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
            
        return {"messages": [AIMessage(content="Tôi đã lưu thông tin danh thiếp vào file data.csv xong.")]}
    else:
        return {"messages": [AIMessage(content="Không có dữ liệu để lưu.")]}

def route_message(state: State) -> Literal["ocr_processing", "__end__"]:
    last_message = state["messages"][-1]
 
    if last_message.tool_calls:
        return "ocr_processing"
    return "__end__"

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_node("ocr_processing", ocr_processing)
workflow.add_node("save_to_csv", save_to_csv)
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    route_message
)
workflow.add_edge("ocr_processing", "save_to_csv")
workflow.add_edge("save_to_csv", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def print_color(text, color="white"):
    colors = {
        "cyan": "\033[96m",   # Màu cho User
        "green": "\033[92m",  # Màu cho LLM trả lời
        "yellow": "\033[93m", # Màu cho Tool/System gọi
        "red": "\033[91m",    # Màu báo lỗi/Chú ý
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def run_and_print(inputs, config):
    print_color("\n" + "="*50, "cyan")
    print_color(f"USER: {inputs['messages'][0].content}", "cyan")
    
    # stream_mode='updates' giúp nhận về kết quả ngay khi một Node chạy xong
    for event in app.stream(inputs, config=config, stream_mode="updates"):
        
        # 1. BẮT SỰ KIỆN TỪ NODE CHATBOT (LLM)
        if "chatbot" in event:
            message = event["chatbot"]["messages"][-1]
            
            # Trường hợp LLM quyết định gọi Tool
            if message.tool_calls:
                print_color(f"LLM (Thinking): Phát hiện hình ảnh, đang gọi tool OCR...", "yellow")
                for tool in message.tool_calls:
                    print_color(f"   -> Tool Call: {tool['name']} | Args: {tool['args']}", "yellow")
            
            # Trường hợp LLM trả lời text bình thường
            else:
                print_color(f"LLM: {message.content}", "green")

        # 2. BẮT SỰ KIỆN TỪ NODE OCR
        if "ocr_processing" in event:
            # Lấy thông tin trích xuất được từ state
            data = event["ocr_processing"].get("extracted_data")
            print_color(f"OCR TOOL: Đã trích xuất dữ liệu thành công!", "yellow")
            print(json.dumps(data, indent=2, ensure_ascii=False)) # In JSON đẹp

        # 3. BẮT SỰ KIỆN TỪ NODE SAVE
        if "save_to_csv" in event:
            message = event["save_to_csv"]["messages"][-1]
            print_color(f"SAVE NODE: {message.content}", "green")

    print_color("="*50 + "\n", "cyan")

def run_chat_session():
    # 1. Cấu hình Memory (Thread ID cố định cho phiên chat này để nhớ lịch sử)
    config = {"configurable": {"thread_id": "terminal_session_01"}}
    
    print("\n" + "="*60)
    print("AI AGENT TERMINAL - OCR ASSISTANT")
    print("------------------------------------------------------------")
    print("Hướng dẫn:")
    print("   - Nhập tin nhắn để trò chuyện.")
    print("   - Nhập 'đường dẫn file ảnh' (ví dụ: card.png) để trích xuất.")
    print("   - Nhấn Ctrl + C để thoát chương trình.")
    print("="*60 + "\n")

    # 2. Khởi tạo System Message (Tùy chọn: Để định hướng bot ngay từ đầu)
    # system_msg = SystemMessage(content="Bạn là trợ lý AI hữu ích. Nếu nhận được ảnh, hãy dùng tool OCR.")
    # app.invoke({"messages": [system_msg]}, config=config)

    try:
        while True:
            try:
                user_input = input("BẠN: ").strip()
                if not user_input: continue
            except EOFError:
                break 

            if os.path.isfile(user_input):
                print(f"\033[90m[Hệ thống: Đã phát hiện file ảnh '{user_input}']\033[0m")
                message_content = f"Hãy xử lý và trích xuất thông tin từ file ảnh tại đường dẫn: '{user_input}'"
            else:
                message_content = user_input

            # Tạo payload gửi vào Graph
            inputs = {"messages": [HumanMessage(content=message_content)]}
            run_and_print(inputs, config)

    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Đã nhận lệnh Ctrl+C. Kết thúc phiên chat.")
        print("Hẹn gặp lại!")
        print("="*60)
        sys.exit(0)

# Cấu hình thread_id
config = {"configurable": {"thread_id": "session_01"}}

# print("TEST 1: CHAT TEXT")
# inputs_1 = {"messages": [HumanMessage(content="Xin chào, hôm nay trời đẹp quá!")]}
# run_and_print(inputs_1, config)

# print("TEST 2: XỬ LÝ ẢNH")
# inputs_2 = {"messages": [HumanMessage(content="Đây là danh thiếp của đối tác, scan thông tin và lưu thông tin giúp tôi: 'C:\code\simple_oop\src\week8\images\card3.jpg'")]}
# run_and_print(inputs_2, config)

run_chat_session()
