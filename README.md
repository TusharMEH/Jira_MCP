# Jira MCP Client with Knowledge Base RAG

An intelligent IT Support assistant that combines OpenAI's language model with a RAG (Retrieval-Augmented Generation) knowledge base and Jira ticketing system. This system provides natural language interaction for IT support queries, automatically searches the knowledge base for solutions, and can create Jira tickets when escalation is needed.

## Overview

This project integrates three main components:

1. **OpenAI LLM** - Provides natural language understanding and generation
2. **IT Support Knowledge Base** - A FAISS-indexed knowledge base built from PDF documentation using sentence transformers
3. **Jira Integration** - Full access to Jira via MCP-Atlassian server for ticket management

The system follows a smart workflow: when users ask IT support questions, it first searches the knowledge base for solutions. If the solution doesn't help or the user wants to escalate, it can automatically create Jira tickets.

## Features

- **Intelligent IT Support**: Natural language queries answered using the knowledge base
- **RAG-Powered Search**: Semantic search through IT support documentation using FAISS and sentence transformers
- **Jira Integration**: Full CRUD operations on Jira issues (create, read, update, delete)
- **Interactive Chat Interface**: Conversational interface for IT support interactions
- **Automatic Ticket Creation**: Escalate unresolved issues to Jira automatically

## Project Structure

```
Jira_MCP/
├── mcp_client_final.py          # Main client application
├── create_index.py               # Knowledge base indexing script
├── run_mcp_server.py             # MCP server launcher
├── requirements.txt              # Python dependencies
├── tools.json                    # Jira tools schema
├── IT_Support_Knowledge_Base.pdf # Source knowledge base PDF
├── kb_faiss_index.bin           # Generated FAISS index (after indexing)
├── kb_chunks.pkl                 # Generated chunks metadata (after indexing)
└── README.md                     # This file
```

## Prerequisites

- Python 3.8 or higher
- Access to a Jira instance (Cloud or Server)
- OpenAI API key
- PDF file containing IT Support Knowledge Base (included as `IT_Support_Knowledge_Base.pdf`)

## Installation

### 1. Clone or Download the Repository

```bash
cd Jira_MCP
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory with the following variables:

- `OPENAI_API_KEY` - Your OpenAI API key
- `OPENAI_MODEL` - OpenAI model to use (default: `gpt-4o-mini`)
- `JIRA_URL` - Your Jira instance URL
- `JIRA_USERNAME` - Your Jira username/email
- `JIRA_API_TOKEN` - Your Jira API token



### 5. Create the Knowledge Base Index

Before running the main application, you need to create the FAISS index from the PDF:

```bash
python create_index.py
```

This script will:
- Extract text from `IT_Support_Knowledge_Base.pdf`
- Chunk the content by section headings
- Create embeddings using sentence transformers
- Build a FAISS index for fast similarity search
- Save the index and chunks to disk

**Output files:**
- `kb_faiss_index.bin` - FAISS vector index
- `kb_chunks.pkl` - Chunks metadata

## Usage

### Running the Main Application

Once the knowledge base is indexed, start the interactive client:

```bash
python mcp_client_final.py
```

The application will:
1. Load the knowledge base index
2. Connect to the MCP-Atlassian server
3. Initialize the OpenAI LLM with available tools
4. Start an interactive chat session

### Example Queries

**Knowledge Base Queries:**
- "How do I change my password?"
- "My screen is frozen, what should I do?"
- "My account is locked"
- "How to connect to VPN from home?"
- "I need to install new software"

**Jira Operations:**
- "Create a ticket: Cannot access email"
- "Get status of IT-1"
- "Search for all open tickets in project IT"


### Interactive Commands

While in the chat interface, you can use these commands:

- `quit` or `exit` - Exit the application
- `tools` - List all available tools
- `clear` - Clear conversation history
- `history` - Show conversation history length

## How It Works

### 1. Knowledge Base Search (RAG)

When a user asks an IT support question:
- The query is embedded using sentence transformers
- FAISS searches for the most relevant sections in the knowledge base
- Top results are retrieved and provided to the LLM
- The LLM formats the answer based on the retrieved context

### 2. Jira Integration

The system connects to Jira via the MCP-Atlassian server, which provides access to:
- Issue creation, reading, updating, and deletion
- Issue search using JQL (Jira Query Language)
- Status transitions
- Comments and worklogs
- Sprint and board management
- Project and version management

### 3. Workflow Logic

The LLM follows this workflow:
1. **For IT support questions**: Search knowledge base first → Provide solution → Ask if it helped → Create ticket if needed
2. **For explicit Jira requests**: Use Jira tools directly
3. **For ticket status**: Fetch and display ticket information

## Available Jira Tools

The system has access to numerous Jira operations including:

- `jira_create_issue` - Create new tickets
- `jira_get_issue` - Get ticket details
- `jira_search` - Search tickets using JQL
- `jira_update_issue` - Update ticket fields
- `jira_transition_issue` - Change ticket status
- `jira_add_comment` - Add comments to tickets
- And many more (see `tools.json` for complete list)

## Troubleshooting

### Knowledge Base Not Found

If you see warnings about missing knowledge base files:
- Ensure you've run `create_index.py` first
- Check that `kb_faiss_index.bin` and `kb_chunks.pkl` exist in the project directory

### MCP Server Connection Issues

- Verify your `.env` file has correct Jira credentials
- Ensure `mcp-atlassian` is installed: `pip install mcp-atlassian`
- Check that your Jira URL is accessible

### OpenAI API Errors

- Verify your `OPENAI_API_KEY` is set correctly in `.env`
- Check your OpenAI account has sufficient credits
- Ensure the API key has proper permissions

## Technical Details

### Embedding Model

- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- Dimension: 384
- Index Type: FAISS IndexFlatL2 (exact L2 distance search)

### Dependencies

Key libraries used:
- `langchain-openai` - OpenAI LLM integration
- `mcp-atlassian` - Jira MCP server
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings
- `pdfplumber` - PDF text extraction

## License

This project is provided as-is for IT support automation purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure environment variables are correctly configured
4. Review error messages for specific guidance
