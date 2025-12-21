"""
Jira MCP Client with OpenAI LLM
Connects local OpenAI LLM to sooperset/mcp-atlassian local MCP server
"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


async def main():
    print("\n" + "="*70)
    print("🤖 OpenAI LLM + Local MCP-Atlassian Integration")
    print("="*70)
    
    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Your Jira Project Key
    JIRA_PROJECT_KEY = "IT"
    
    
    # ==================================================================
    # STEP 1: Connect to Local MCP-Atlassian Server
    # ==================================================================
    print("\n=== STEP 1: Connecting to MCP-Atlassian Server ===")
    print(f"Jira URL: {os.getenv('JIRA_URL')}")
    print(f"Default Project: {JIRA_PROJECT_KEY}")
    
    server_params = StdioServerParameters(
        command="mcp-atlassian",
        args=["--env-file", ".env", "-vv"],
        env={**os.environ}
    )
    
    print("Launching mcp-atlassian server...")
    
    async with stdio_client(server_params) as (read, write):
        print("✓ mcp-atlassian server started")
        
        async with ClientSession(read, write) as session:
            print("✓ MCP session created")
            
            await session.initialize()
            print("✓ Handshake completed")
            
            
            # ==================================================================
            # STEP 2: Discover Jira Tools
            # ==================================================================
            print("\n=== STEP 2: Discovering Jira Tools ===")
            
            tools_response = await session.list_tools()
            
            print(f"\n✓ Found {len(tools_response.tools)} Jira tools:")
            for i, tool in enumerate(tools_response.tools, 1):
                print(f"{i:2d}. {tool.name}")
            
            
            # ==================================================================
            # STEP 3: Setup OpenAI LLM
            # ==================================================================
            print("\n=== STEP 3: Setting up OpenAI LLM ===")
            
            if not OPENAI_API_KEY:
                print("\n⚠️  ERROR: Please set OPENAI_API_KEY in .env file!")
                return
            
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
                temperature=0
            )
            print(f"✓ OpenAI LLM initialized (model: {OPENAI_MODEL})")
            
            # Convert MCP tools to LangChain format
            print("\nConverting MCP tools to LangChain format...")
            langchain_tools = []
            for tool in tools_response.tools:
                langchain_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            llm_with_tools = llm.bind_tools(langchain_tools)
            print(f"✓ OpenAI now has access to {len(langchain_tools)} Jira tools!")
            
            
            # ==================================================================
            # STEP 4: Interactive Chat with Jira
            # ==================================================================
            print("\n=== STEP 4: Chat with Jira via OpenAI ===")
            print("\n💡 Example queries you can try:")
            print(f"   • 'List all my Jira projects'")
            print(f"   • 'Show me open issues in project {JIRA_PROJECT_KEY}'")
            print(f"   • 'Create a task: Login page not working'")
            print(f"   • 'Change priority of {JIRA_PROJECT_KEY}-1 to High'")
            print(f"   • 'Add comment to {JIRA_PROJECT_KEY}-1: Working on this'")
            print("\nType 'quit' to exit, 'tools' to list available tools\n")
            
            SYSTEM_PROMPT = f"""You are a helpful AI assistant with access to Jira tools.

JIRA CONFIGURATION:
- Default project key: {JIRA_PROJECT_KEY}
- Always use project_key: "{JIRA_PROJECT_KEY}" when creating issues unless user specifies otherwise
- Valid issue types: Task, Epic, Subtask (use Task if unsure)

JIRA TOOL USAGE:
- Only use Jira tools when the user explicitly asks to create, update, search, or manage Jira issues/projects
- When creating issues, always use project_key: "{JIRA_PROJECT_KEY}"
- For any incident, bug, problem, or request → use issue_type: "Task"
- For updating issues (priority, status, etc.) → use jira_update_issue directly without fetching first
- Valid priorities: Highest, High, Medium, Low, Lowest

ERROR HANDLING:
- If a tool returns an error, explain the error to the user in simple terms
- Suggest what might have gone wrong and how to fix it
- Never leave the user without a response

NORMAL CONVERSATION:
- For greetings (hi, hello, thanks) → respond naturally without using tools
- For general questions → answer directly without tools

Only call Jira tools when user clearly wants to interact with Jira."""
            
            while True:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'tools':
                    print("\n📋 Available Jira Tools:")
                    for i, tool in enumerate(tools_response.tools, 1):
                        print(f"{i:2d}. {tool.name}")
                    continue
                
                if not user_input:
                    continue
                
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_input)
                ]
                
                print("\n🤖 OpenAI thinking...")
                try:
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)
                    
                    # Loop to handle multiple rounds of tool calls
                    max_iterations = 5
                    iteration = 0
                    
                    while response.tool_calls and iteration < max_iterations:
                        iteration += 1
                        print(f"\n🔧 OpenAI calling Jira tools (round {iteration}):\n")
                        
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            print(f"  📞 Tool: {tool_name}")
                            print(f"  📝 Args: {tool_args}")
                            print(f"  ⏳ Executing...", end=" ", flush=True)
                            
                            try:
                                result = await session.call_tool(tool_name, tool_args)
                                
                                tool_result_text = ""
                                if result.content:
                                    for content_item in result.content:
                                        if hasattr(content_item, 'text'):
                                            tool_result_text += content_item.text
                                
                                if "error" in tool_result_text.lower():
                                    print("⚠️ Error")
                                else:
                                    print("✓")
                                
                                if tool_result_text:
                                    preview = tool_result_text[:300].replace('\n', ' ')
                                    print(f"  📄 Result: {preview}\n")
                                else:
                                    tool_result_text = "Tool returned empty result."
                                
                            except Exception as tool_error:
                                print("❌ Failed")
                                tool_result_text = f"Error: {str(tool_error)}"
                            
                            messages.append(
                                ToolMessage(
                                    content=tool_result_text,
                                    tool_call_id=tool_call["id"]
                                )
                            )
                        
                        # Get next response
                        print("🤖 OpenAI processing...")
                        response = llm_with_tools.invoke(messages)
                        messages.append(response)
                    
                    # Final response
                    print("\n💬 OpenAI:")
                    print("-" * 70)
                    if hasattr(response, 'content') and response.content:
                        print(response.content)
                    else:
                        print("Operation completed.")
                    print("-" * 70)
                        
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    print("Please try again or type 'quit' to exit.")
            
            
            print("\n=== Session Complete ===")
            print("✓ Disconnecting from MCP-Atlassian")


if __name__ == "__main__":
    print("\n📋 Prerequisites Check:")
    print("  ✓ mcp-atlassian installed (pip)")
    print("  ✓ .env file with Jira credentials")
    print("  ? OPENAI_API_KEY in .env file")
    print("\nStarting...\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
