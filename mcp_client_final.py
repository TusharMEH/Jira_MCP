"""
Jira MCP Client with OpenAI LLM + Knowledge Base RAG
Connects OpenAI LLM to:
1. IT Support Knowledge Base (FAISS + sentence-transformers)
2. Jira via sooperset/mcp-atlassian MCP server
"""

import asyncio
import os
import pickle                                          # ← NEW: For loading chunks
import faiss                                           # ← NEW: For FAISS index
import numpy as np                                     # ← NEW: For embeddings
from sentence_transformers import SentenceTransformer # ← NEW: For encoding queries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# NEW SECTION: Knowledge Base Configuration & Functions
# ============================================================================

# KB Configuration
KB_INDEX_PATH = "kb_faiss_index.bin"
KB_CHUNKS_PATH = "kb_chunks.pkl"
KB_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Global variables for KB (loaded once)
kb_index = None
kb_chunks = None
kb_model = None


def load_knowledge_base():
    """Load FAISS index, chunks, and embedding model"""
    global kb_index, kb_chunks, kb_model
    
    print("\n📚 Loading Knowledge Base...")
    
    # Check if files exist
    if not os.path.exists(KB_INDEX_PATH):
        print(f"   ⚠️  KB index not found: {KB_INDEX_PATH}")
        return False
    if not os.path.exists(KB_CHUNKS_PATH):
        print(f"   ⚠️  KB chunks not found: {KB_CHUNKS_PATH}")
        return False
    
    # Load FAISS index
    kb_index = faiss.read_index(KB_INDEX_PATH)
    print(f"   ✓ FAISS index loaded: {kb_index.ntotal} vectors")
    
    # Load chunks
    with open(KB_CHUNKS_PATH, 'rb') as f:
        kb_chunks = pickle.load(f)
    print(f"   ✓ Chunks loaded: {len(kb_chunks)} sections")
    
    # Load embedding model
    kb_model = SentenceTransformer(KB_EMBEDDING_MODEL)
    print(f"   ✓ Embedding model loaded: {KB_EMBEDDING_MODEL}")
    
    return True


def search_knowledge_base(query: str, top_k: int = 2) -> list:
    """
    Search the IT Support Knowledge Base for relevant information.
    
    Args:
        query: The user's question
        top_k: Number of results to return
        
    Returns:
        List of relevant KB sections with scores
    """
    global kb_index, kb_chunks, kb_model
    
    if kb_index is None or kb_chunks is None or kb_model is None:
        return []
    
    # Encode query
    query_embedding = kb_model.encode([query]).astype('float32')
    
    # Search FAISS
    distances, indices = kb_index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(kb_chunks):
            chunk = kb_chunks[idx]
            score = 1 / (1 + dist)  # Convert distance to similarity score
            results.append({
                "title": chunk["title"],
                "content": chunk["content"],
                "score": score,
                "section_num": chunk["section_num"]
            })
    
    return results


# ============================================================================
# NEW: KB Retriever Tool Definition (for LangChain)
# ============================================================================

def create_kb_retriever_tool():
    """
    Create a tool definition for the KB retriever that LangChain/OpenAI can use.
    This follows the same format as MCP tools.
    """
    return {
        "type": "function",
        "function": {
            "name": "search_it_knowledge_base",
            "description": (
                "Search the IT Support Knowledge Base for solutions to common IT problems. "
                "Use this tool FIRST before creating Jira tickets. "
                "The KB contains guides for: password changes, password resets, frozen screens, "
                "account lockouts, VPN connection, and software installation requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's IT support question or problem description"
                    }
                },
                "required": ["query"]
            }
        }
    }


# ============================================================================
# NEW: Execute KB Tool (handles the actual search)
# ============================================================================

def execute_kb_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute the KB retriever tool and return results as formatted string.
    """
    if tool_name == "search_it_knowledge_base":
        query = tool_args.get("query", "")
        results = search_knowledge_base(query, top_k=2)
        
        if not results:
            return "No relevant information found in the IT Knowledge Base."
        
        # Format results for LLM
        output = "📚 **IT Knowledge Base Results:**\n\n"
        for i, result in enumerate(results, 1):
            output += f"**{i}. {result['title']}** (relevance: {result['score']:.2f})\n"
            output += f"{result['content']}\n\n"
            output += "-" * 50 + "\n\n"
        
        return output
    
    return f"Unknown KB tool: {tool_name}"


# ============================================================================
# END OF NEW KB SECTION
# ============================================================================


async def main():
    print("\n" + "="*70)
    print("🤖 OpenAI LLM + Knowledge Base + Jira MCP Integration")  # ← CHANGED: Updated title
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
    # NEW STEP: Load Knowledge Base
    # ==================================================================
    kb_loaded = load_knowledge_base()
    if not kb_loaded:
        print("   ⚠️  Knowledge Base not available - will only use Jira tools")
    
    
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
            
            # ============================================================
            # NEW: Add KB Retriever Tool FIRST (so it's prioritized)
            # ============================================================
            if kb_loaded:
                kb_tool = create_kb_retriever_tool()
                langchain_tools.append(kb_tool)
                print("✓ Added: search_it_knowledge_base (KB Retriever)")
            # ============================================================
            
            # Add MCP Jira tools
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
            print(f"✓ OpenAI now has access to {len(langchain_tools)} tools!")  # ← CHANGED: says "tools" not "Jira tools"
            
            
            # ==================================================================
            # STEP 4: Interactive Chat
            # ==================================================================
            print("\n=== STEP 4: Chat with IT Support + Jira ===")  # ← CHANGED
            print("\n💡 Example queries you can try:")
            print("   • 'How do I change my password?' (KB)")           # ← NEW
            print("   • 'My screen is frozen, what should I do?' (KB)") # ← NEW
            print("   • 'My account is locked' (KB)")                    # ← NEW
            print(f"   • 'Create a ticket: Cannot access email' (Jira)")
            print(f"   • 'Get status of {JIRA_PROJECT_KEY}-1' (Jira)")
            print("\nType 'quit' to exit, 'tools' to list tools, 'clear' to reset history\n")
            
            # ============================================================
            # CHANGED: Updated System Prompt for KB + Jira workflow
            # ============================================================
            SYSTEM_PROMPT = f"""You are a helpful IT Support assistant with access to:
1. IT Support Knowledge Base (search_it_knowledge_base tool)
2. Jira ticketing system (jira_* tools)

CRITICAL WORKFLOW - YOU MUST FOLLOW THIS ORDER:

1. For ANY IT support question (password, frozen screen, VPN, software, account issues, 
   hardware requests, laptop, equipment, access problems, etc.):
   → ALWAYS use search_it_knowledge_base FIRST - do not answer from your own knowledge
   → Provide the solution from the Knowledge Base results
   → Ask if the solution helped
   → If user says NO or wants to escalate → create a Jira ticket using jira_create_issue

2. For explicit Jira requests ("create ticket", "check status", "update issue"):
   → Use the appropriate jira_* tool directly

3. For ticket status inquiries (e.g., "what's the status of IT-5"):
   → Use jira_get_issue to fetch the ticket details

4. When user confirms they want a ticket (says "yes", "please", "create one"):
   → Remember the previous context and create a Jira ticket with that information
   → Use jira_create_issue with project_key: "{JIRA_PROJECT_KEY}"

JIRA CONFIGURATION:
- Default project key: {JIRA_PROJECT_KEY}
- Valid issue types: Task, Epic, Subtask (use Task for support requests)
- When creating tickets, include the original problem in the description

RESPONSE STYLE:
- Be helpful and friendly
- When providing KB solutions, format them clearly with steps
- After providing a KB solution, always ask: "Did this solve your issue? If not, I can create a support ticket for you."
- For greetings only (hi, hello), respond naturally without using any tools
- For ANY question about IT topics, USE THE KB TOOL FIRST

ERROR HANDLING:
- If KB search returns no relevant results, say so and offer to create a Jira ticket
- If a Jira tool fails, explain the error simply"""
            # ============================================================
            
            # ============================================================
            # FIXED: Initialize messages list ONCE before the loop
            # This maintains conversation history across turns
            # ============================================================
            messages = [
                SystemMessage(content=SYSTEM_PROMPT)
            ]
            
            while True:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'tools':
                    print("\n📋 Available Tools:")
                    print("\n  📚 Knowledge Base:")
                    print("      1. search_it_knowledge_base")
                    print("\n  📝 Jira Tools:")
                    for i, tool in enumerate(tools_response.tools, 1):
                        print(f"     {i+1:2d}. {tool.name}")
                    continue
                
                # ← NEW: Clear conversation history
                if user_input.lower() == 'clear':
                    messages = [SystemMessage(content=SYSTEM_PROMPT)]
                    print("\n🗑️  Conversation history cleared!")
                    continue
                
                # ← NEW: Show history count
                if user_input.lower() == 'history':
                    print(f"\n📜 Conversation has {len(messages)} messages (including system prompt)")
                    continue
                
                if not user_input:
                    continue
                
                # Append new user message to existing conversation history
                messages.append(HumanMessage(content=user_input))
                
                print("\n🤖 Processing...")
                try:
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)
                    
                    # Loop to handle multiple rounds of tool calls
                    max_iterations = 5
                    iteration = 0
                    
                    while response.tool_calls and iteration < max_iterations:
                        iteration += 1
                        print(f"\n🔧 Calling tools (round {iteration}):\n")
                        
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            print(f"  📞 Tool: {tool_name}")
                            print(f"  📝 Args: {tool_args}")
                            print(f"  ⏳ Executing...", end=" ", flush=True)
                            
                            try:
                                # ============================================
                                # NEW: Check if it's KB tool or MCP tool
                                # ============================================
                                if tool_name == "search_it_knowledge_base":
                                    # Execute KB tool locally
                                    tool_result_text = execute_kb_tool(tool_name, tool_args)
                                    print("✓ (KB)")
                                else:
                                    # Execute MCP/Jira tool
                                    result = await session.call_tool(tool_name, tool_args)
                                    
                                    tool_result_text = ""
                                    if result.content:
                                        for content_item in result.content:
                                            if hasattr(content_item, 'text'):
                                                tool_result_text += content_item.text
                                    
                                    if "error" in tool_result_text.lower():
                                        print("⚠️ Error (Jira)")
                                    else:
                                        print("✓ (Jira)")
                                # ============================================
                                
                                if tool_result_text:
                                    # Show preview (shorter for KB results since they're long)
                                    if tool_name == "search_it_knowledge_base":
                                        preview = tool_result_text[:200].replace('\n', ' ')
                                    else:
                                        preview = tool_result_text[:300].replace('\n', ' ')
                                    print(f"  📄 Result: {preview}...\n")
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
                        print("🤖 Processing results...")
                        response = llm_with_tools.invoke(messages)
                        messages.append(response)
                    
                    # Final response
                    print("\n💬 Assistant:")
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
    print("  ? kb_faiss_index.bin (Knowledge Base)")      # ← NEW
    print("  ? kb_chunks.pkl (Knowledge Base)")           # ← NEW
    print("\nStarting...\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()