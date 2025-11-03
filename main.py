"""
Jira MCP Client with Groq LLM
Connects local Groq LLM to Storm AI's Jira MCP server
"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()


async def main():
    print("\n" + "="*70)
    print("🤖 Groq LLM + Storm AI Jira Integration")
    print("="*70)
    
    # ==================================================================
    # CONFIGURATION - Update these values
    # ==================================================================
    
    # Storm AI Configuration
    STORM_GATEWAY_URL = os.getenv("STORM_GATEWAY_URL")
    STORM_API_KEY = os.getenv("STORM_API_KEY")
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
    # GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"     # You can change the model
    
    # Path to npx (same as Claude Desktop config)
    NPX_PATH = "C:\\Program Files\\nodejs\\npx"   # change as per yours
    
    
    # ==================================================================
    # STEP 1: Connect to Storm AI MCP Server
    # ==================================================================
    print("\n=== STEP 1: Connecting to Storm AI ===")
    print(f"Gateway: {STORM_GATEWAY_URL}")
    
    # Configure mcp-remote bridge (same as Claude Desktop)
    server_params = StdioServerParameters(
        command=NPX_PATH,
        args=[
            "-y",                    # Auto-install mcp-remote if needed
            "mcp-remote",            # The bridge tool
            STORM_GATEWAY_URL,       # Storm AI endpoint
            "--header",              # Authentication
            f"X-API-Key: {STORM_API_KEY}"
        ],
        env=None
    )
    
    print("Launching mcp-remote bridge...")
    
    async with stdio_client(server_params) as (read, write):
        print("✓ mcp-remote bridge started")
        print("✓ Connected to Storm AI gateway")
        
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
            
            print(f"\n✓ Found {len(tools_response.tools)} Jira tools from Storm AI:")
            for i, tool in enumerate(tools_response.tools, 1):
                print(f"{i:2d}. {tool.name}")
                # Print first 80 chars of description
                desc = tool.description.replace('\n', ' ')[:80]
                print(f"    {desc}...")
            
            
            # ==================================================================
            # STEP 3: Setup Groq LLM
            # ==================================================================
            print("\n=== STEP 3: Setting up Groq LLM ===")
            
            if GROQ_API_KEY == "your_groq_api_key_here":
                print("\n⚠️  ERROR: Please set your Groq API key in the script!")
                print("Get it from: https://console.groq.com/keys")
                return
            
            # Initialize Groq
            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0
            )
            print(f"✓ Groq LLM initialized (model: {GROQ_MODEL})")
            
            # Convert Storm AI tools to LangChain format
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
            
            
            # ==================================================================
            # STEP 4: Interactive Chat with Jira
            # ==================================================================
            print("\n=== STEP 4: Chat with Jira via Groq ===")
            print("\n💡 Example queries you can try:")
            print("   • 'List all my Jira projects'")
            print("   • 'Show me open issues in project PROJ'")
            print("   • 'Create a bug: Login page not working'")
            print("   • 'Get details for issue PROJ-123'")
            print("   • 'Search for issues with high priority'")
            print("   • 'Add comment to PROJ-456: Working on this'")
            print("\nType 'quit' to exit, 'tools' to list available tools\n")
            
            
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
                
                from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

                # Add this BEFORE the while loop (around line ~170)
                SYSTEM_PROMPT = """You are a helpful AI assistant with access to Jira tools.

                JIRA TOOL USAGE:
                - Only use Jira tools when the user explicitly asks to create, update, search, or manage Jira issues/projects
                - Valid issue types: Task only
                - For "incident" or "security breach" → use Task type with High priority

                NORMAL CONVERSATION:
                - For greetings (hi, hello, thanks) → respond naturally without using tools
                - For general questions (what is ML, explain concepts) → answer directly without tools
                - For casual chat → be friendly and helpful

                Only call Jira tools when user clearly wants to interact with Jira."""

                # In the chat loop, change to:
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
                        
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            print(f"  📞 Tool: {tool_name}")
                            print(f"  📝 Args: {tool_args}")
                            print(f"  ⏳ Executing via Storm AI...", end=" ", flush=True)
                            
                            # Call Storm AI's Jira tool via MCP
                            result = await session.call_tool(tool_name, tool_args)
                            
                            # Parse the result properly - MCP returns content as a list
                            # This follows the official MCP specification
                            tool_result_text = ""
                            if result.content:
                                for content_item in result.content:
                                    if hasattr(content_item, 'text'):
                                        tool_result_text += content_item.text
                            
                            print("✓")
                            print(f"  📊 Result: {len(tool_result_text)} characters received")
                            
                            # Debug: Show first 200 chars of result
                            if tool_result_text:
                                # preview = tool_result_text[:200].replace('\n', ' ')
                                preview = tool_result_text.replace('\n', ' ')
                                print(f"  📄 Preview: {preview}...\n")
                            else:
                                print(f"  ⚠️  Warning: Empty result received\n")
                            
                            # Add result to messages
                            messages.append(
                                ToolMessage(
                                    content=tool_result_text if tool_result_text else "Tool returned empty result",
                                    tool_call_id=tool_call["id"]
                                )
                            )
                        
                        # Get final response from Groq with tool results
                        print("🤖 Groq processing results...\n")
                        final_response = llm_with_tools.invoke(messages)
                        
                        # Display the final response
                        print("💬 Groq:")
                        print("-" * 70)
                        if hasattr(final_response, 'content') and final_response.content:
                            print(final_response.content)
                        else:
                            print("(No response text generated)")
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
            print("✓ Disconnecting from Storm AI")


if __name__ == "__main__":
    print("\n📋 Prerequisites Check:")
    print("  ✓ Storm AI working in Claude Desktop")
    print("  ✓ Node.js installed (npx available)")
    print("  ? Groq API key configured in script")
    print("\nStarting...\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure Storm AI is working in Claude Desktop first!")