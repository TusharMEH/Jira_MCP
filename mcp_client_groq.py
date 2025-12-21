"""
Jira MCP Client with Groq LLM
Connects local Groq LLM to sooperset/mcp-atlassian local MCP server
"""

import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


async def main():
    print("\n" + "="*70)
    print("🤖 Groq LLM + Local MCP-Atlassian Integration")
    print("="*70)
    
    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
    GROQ_MODEL = "openai/gpt-oss-120b"
    
    # Your Jira Project Key
    JIRA_PROJECT_KEY = "IT"  # <-- Your project key
    
    
    # ==================================================================
    # STEP 1: Connect to Local MCP-Atlassian Server
    # ==================================================================
    print("\n=== STEP 1: Connecting to MCP-Atlassian Server ===")
    print(f"Jira URL: {os.getenv('JIRA_URL')}")
    print(f"Default Project: {JIRA_PROJECT_KEY}")
    
    # Configure local mcp-atlassian server
    server_params = StdioServerParameters(
        command="mcp-atlassian",
        args=["--env-file", ".env","-vv"],
        env={**os.environ}
    )
    
    print("Launching mcp-atlassian server...")
    
    async with stdio_client(server_params) as (read, write):
        print("✓ mcp-atlassian server started")
        
        async with ClientSession(read, write) as session:
            print("✓ MCP session created")
            
            # Initialize the session (handshake)
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
            # STEP 3: Setup Groq LLM
            # ==================================================================
            print("\n=== STEP 3: Setting up Groq LLM ===")
            
            if not GROQ_API_KEY:
                print("\n⚠️  ERROR: Please set GROQ_API_KEY in .env file!")
                print("Get it from: https://console.groq.com/keys")
                return
            
            # Initialize Groq
            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0
            )
            print(f"✓ Groq LLM initialized (model: {GROQ_MODEL})")
            
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
            
            # Bind tools to Groq
            llm_with_tools = llm.bind_tools(langchain_tools)
            print(f"✓ Groq now has access to {len(langchain_tools)} Jira tools!")
            
            with open("tools.json","w") as fp:
                json.dump(langchain_tools,fp)
                
            
            # ==================================================================
            # STEP 4: Interactive Chat with Jira
            # ==================================================================
            print("\n=== STEP 4: Chat with Jira via Groq ===")
            print("\n💡 Example queries you can try:")
            print(f"   • 'List all my Jira projects'")
            print(f"   • 'Show me open issues in project {JIRA_PROJECT_KEY}'")
            print(f"   • 'Create a task: Login page not working'")
            print(f"   • 'Get details for issue {JIRA_PROJECT_KEY}-1'")
            print(f"   • 'Search for issues with high priority'")
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
- For large features or initiatives → use issue_type: "Epic"
- For sub-items of a parent issue → use issue_type: "Subtask"

ERROR HANDLING:
- If a tool returns an error, explain the error to the user in simple terms
- Suggest what might have gone wrong and how to fix it
- Never leave the user without a response

NORMAL CONVERSATION:
- For greetings (hi, hello, thanks) → respond naturally without using tools
- For general questions → answer directly without tools
- For casual chat → be friendly and helpful

Only call Jira tools when user clearly wants to interact with Jira."""
            
            while True:
                # Get user input
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
                
                print("\n🤖 Groq thinking...")
                try:
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)
                    
                    # Check if Groq wants to use Jira tools
                    if response.tool_calls:
                        print("\n🔧 Groq decided to call Jira tools:\n")
                        
                        tool_had_error = False
                        
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            print(f"  📞 Tool: {tool_name}")
                            print(f"  📝 Args: {tool_args}")
                            print(f"  ⏳ Executing via MCP-Atlassian...", end=" ", flush=True)
                            
                            try:
                                # Call mcp-atlassian tool via MCP
                                result = await session.call_tool(tool_name, tool_args)
                                
                                # Parse the result - MCP returns content as a list
                                tool_result_text = ""
                                if result.content:
                                    for content_item in result.content:
                                        if hasattr(content_item, 'text'):
                                            tool_result_text += content_item.text
                                
                                # Check if result contains an error
                                if "error" in tool_result_text.lower():
                                    print("⚠️ Error")
                                    tool_had_error = True
                                else:
                                    print("✓")
                                
                                print(f"  📊 Result: {len(tool_result_text)} characters received")
                                
                                if tool_result_text:
                                    preview = tool_result_text[:500].replace('\n', ' ')
                                    print(f"  📄 Response: {preview}\n")
                                else:
                                    print(f"  ⚠️  Warning: Empty result received\n")
                                    tool_result_text = "Tool returned empty result. Please try again or check your Jira configuration."
                                
                            except Exception as tool_error:
                                print("❌ Failed")
                                tool_result_text = f"Error executing tool: {str(tool_error)}"
                                tool_had_error = True
                                print(f"  ❌ Error: {tool_result_text}\n")
                            
                            # Add result to messages
                            messages.append(
                                ToolMessage(
                                    content=tool_result_text,
                                    tool_call_id=tool_call["id"]
                                )
                            )
                        
                        # Get final response from Groq with tool results
                        print("🤖 Groq processing results...\n")
                        final_response = llm_with_tools.invoke(messages)
                        
                        print("💬 Groq:")
                        print("-" * 70)
                        if hasattr(final_response, 'content') and final_response.content:
                            print(final_response.content)
                        else:
                            # Fallback response if Groq doesn't generate content
                            if tool_had_error:
                                print("I encountered an error while trying to perform that action.")
                                print("Please check your Jira project key and try again.")
                            else:
                                print("Operation completed, but I couldn't generate a summary.")
                        print("-" * 70)
                        
                    else:
                        # No tools needed, direct response
                        print("\n💬 Groq:")
                        print("-" * 70)
                        print(response.content)
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
    print("  ? GROQ_API_KEY in .env file")
    print("\nStarting...\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()