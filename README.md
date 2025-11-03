## Jira MCP Client with Groq LLM

Integrates a local Groq LLM (via LangChain) with Storm AI's Jira MCP server through an MCP stdio bridge. This lets you chat with Groq and have it call real Jira tools exposed by Storm AI.

### How it works (project flow)
- **Startup**: The script launches the MCP bridge using `npx mcp-remote` pointing at your Storm AI gateway with `X-API-Key` auth.
- **Handshake**: An MCP `ClientSession` initializes and lists available Jira tools published by Storm AI.
- **LLM setup**: Groq is initialized (`langchain_groq.ChatGroq`) and bound to the discovered MCP tools (converted to OpenAI function-call schema for LangChain).
- **Chat loop**: Your prompt is sent to Groq. If the model decides to use a Jira tool, the script executes it via the MCP session, collects the tool's result, and feeds it back to Groq for the final answer. If no tool is needed, Groq replies directly.

### Prerequisites
- **Storm AI Jira MCP server** running and reachable via a gateway URL.
- **API keys**:
  - `STORM_API_KEY` (aka Storm MCP API key)
  - `GROQ_API_KEY`
- **Gateway URL**:
  - `STORM_GATEWAY_URL` (e.g., `https://gateway.storm.dev/mcp/jira` or as provided by your setup)
- **Node.js** with `npx` available (Windows path commonly `C:\\Program Files\\nodejs\\npx`).
- **Python 3.10+** and `pip`.

### Environment variables
Create a `.env` file in the project root:

```dotenv
STORM_GATEWAY_URL=https://your-storm-gateway.example/mcp/jira
STORM_API_KEY=your_storm_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Notes:
- On Windows, keep the `NPX_PATH` in `main.py` aligned to your Node.js install if different.
- You may change the Groq model in `main.py` (see `GROQ_MODEL`).

### Install
```bash
pip install -r requirements.txt
```

If you use `uv`:
```bash
uv sync
```

### Run
```bash
python main.py
```

At startup the script prints prerequisite checks, connects to Storm AI via MCP, lists available Jira tools, initializes Groq, and enters an interactive chat.

### Example usage (interactive)
- Example queries to type after it starts:
  - "List all my Jira projects"
  - "Show me open issues in project PROJ"
  - "Create a task: Incident - API latency spike, set priority High"
  - "Get details for issue PROJ-123"
  - "Add comment to PROJ-456: Working on this"

Sample session excerpt:
```text
=== STEP 1: Connecting to Storm AI ===
Gateway: https://your-storm-gateway.example/mcp/jira
✓ mcp-remote bridge started
✓ Connected to Storm AI gateway
✓ MCP session created
✓ Handshake completed

=== STEP 2: Discovering Jira Tools ===
✓ Found 12 Jira tools from Storm AI:
  1. jira_search_issues
  2. jira_create_issue
  ...

=== STEP 3: Setting up Groq LLM ===
✓ Groq LLM initialized (model: meta-llama/llama-4-maverick-17b-128e-instruct)
✓ Groq now has access to 12 Jira tools!

=== STEP 4: Chat with Jira via Groq ===

👤 You: Show me open issues in project PROJ
🤖 Groq thinking...
🔧 Groq decided to call Jira tools:
  📞 Tool: jira_search_issues
  📝 Args: {"jql": "project = PROJ AND statusCategory != Done ORDER BY updated DESC"}
  ⏳ Executing via Storm AI... ✓
  📊 Result: 1820 characters received

💬 Groq:
- 3 open issues found in PROJ:
  - PROJ-123 (High): API timeout on login
  - PROJ-125 (Medium): Dashboard chart misrender
  - PROJ-130 (Low): Typo in settings label
```

### Configuration details
- `main.py`
  - Reads `STORM_GATEWAY_URL`, `STORM_API_KEY`, `GROQ_API_KEY` from environment.
  - Uses `NPX_PATH` to launch `mcp-remote` with `--header X-API-Key: <STORM_API_KEY>`.
  - Binds discovered MCP tools to Groq for function calling.
- `jira_mcp_client.py`
  - Earlier variant with a simpler message flow; kept for reference.

### Troubleshooting
- "Please set your Groq API key": Ensure `GROQ_API_KEY` is present in `.env` or environment.
- "Cannot start mcp-remote": Verify Node.js is installed and `NPX_PATH` in `main.py` is correct, or ensure `npx` is on PATH.
- "Handshake failed" or empty tool list: Confirm `STORM_GATEWAY_URL` is correct and `STORM_API_KEY` has access.
- Tool result is empty: The MCP server may return no content for some queries; try a simpler query or verify Jira connectivity in Storm AI.

### Security
- .env  ignored by Git via `.gitignore`.


 ### Additional examples
 - Ask for status and assignee of a specific issue:
   - "What is the status of IT-6 and who is assigned to the issue?"
 
   Example outcome:
   ```text
   🔧 Groq decided to call Jira tools:
     📞 Tool: jira_get_issue
     📝 Args: {"issueIdOrKey": "IT-6"}
     ⏳ Executing via Storm AI... ✓

   💬 Groq:
   - IT-6 status: In Progress
   - Assignee: Jane Doe (@jane.doe)
   ```

 - Create a new issue with title and description:
   - "Create a task in project IT with title 'Login 2FA failure' and description '2FA step intermittently fails for SSO users; capture logs and add retry.'"

   Example outcome:
   ```text
   🔧 Groq decided to call Jira tools:
     📞 Tool: jira_create_issue
     📝 Args: {"projectKey": "IT", "summary": "Login 2FA failure", "description": "2FA step intermittently fails for SSO users; capture logs and add retry.", "issuetype": "Task", "priority": "High"}
     ⏳ Executing via Storm AI... ✓

   💬 Groq:
   - Created issue IT-142 with priority High
   - Link: https://your-jira.example/browse/IT-142
   ```

