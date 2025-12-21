# MCP Project - IT Support Assistant with Knowledge Base & Jira Integration

An intelligent IT Support assistant that combines OpenAI LLM with a RAG (Retrieval-Augmented Generation) knowledge base and Jira ticketing system integration. The system can answer IT support questions from a knowledge base and create/manage Jira tickets when needed.

## Overview

This project consists of two main components:

1. **`mcp_client_final.py`** - Main application that provides an interactive chat interface for IT support
2. **`create_index.py`** - Utility script to create a searchable index from the IT Support Knowledge Base PDF

## Features

- 🤖 **OpenAI LLM Integration** - Powered by GPT models for natural language understanding
- 📚 **Knowledge Base RAG** - Semantic search through IT Support documentation using FAISS and sentence-transformers
- 🎫 **Jira Integration** - Create, update, and query Jira tickets via MCP (Model Context Protocol)
- 💬 **Interactive Chat Interface** - Natural conversation flow for IT support queries
- 🔍 **Intelligent Workflow** - Automatically searches knowledge base first before creating tickets

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Jira account with API token
- `IT_Support_Knowledge_Base.pdf` file in the project directory

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - `mcp-atlassian` - Jira MCP server integration
   - `langchain-openai` - OpenAI LLM integration
   - `pdfplumber` - PDF text extraction
   - `faiss-cpu` - Vector similarity search
   - `sentence-transformers` - Text embeddings
   - `python-dotenv` - Environment variable management

3. **Install MCP Atlassian server:**
   ```bash
   pip install mcp-atlassian
   ```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Jira Configuration
JIRA_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your_email@example.com
JIRA_API_TOKEN=your_jira_api_token
```

### Getting Jira API Token

1. Go to [Atlassian Account Settings](https://id.atlassian.com/manage-profile/security/api-tokens)
2. Click "Create API token"
3. Copy the token and add it to your `.env` file

## Usage

### Step 1: Create the Knowledge Base Index

Before running the main application, you need to create a searchable index from the PDF:

```bash
python create_index.py
```

This script will:
- Extract text from `IT_Support_Knowledge_Base.pdf`
- Chunk the document by section headings
- Create embeddings using `all-MiniLM-L6-v2` model
- Build a FAISS index for fast similarity search
- Save the index (`kb_faiss_index.bin`) and chunks metadata (`kb_chunks.pkl`)

**Output files:**
- `kb_faiss_index.bin` - FAISS vector index
- `kb_chunks.pkl` - Chunks metadata with titles and content

**Note:** You only need to run this once, or whenever you update the PDF.

### Step 2: Run the Main Application

Start the interactive IT Support assistant:

```bash
python mcp_client_final.py
```

The application will:
1. Load the knowledge base index
2. Connect to the MCP-Atlassian server
3. Initialize OpenAI LLM with access to both KB and Jira tools
4. Start an interactive chat session

### Example Queries

Once the application is running, you can ask questions like:

**Knowledge Base Queries:**
- "How do I change my password?"
- "My screen is frozen, what should I do?"
- "My account is locked"
- "How to connect to VPN from home?"
- "I need to install new software"

**Jira Operations:**
- "Create a ticket: Cannot access email"
- "Get status of IT-1"
- "Update ticket IT-5 with comment: Issue resolved"

**Interactive Commands:**
- `quit` or `exit` - Exit the application
- `tools` - List all available tools
- `clear` - Clear conversation history
- `history` - Show conversation history count

## How It Works

### Knowledge Base Search Flow

1. User asks an IT support question
2. System searches the knowledge base using semantic similarity
3. Returns relevant sections from the PDF
4. LLM formats the response based on KB results
5. If solution doesn't help, user can request a Jira ticket

### Jira Integration Flow

1. User explicitly requests Jira operations OR confirms they need a ticket
2. System uses MCP tools to interact with Jira
3. Creates/updates/queries tickets as requested
4. Returns ticket information to the user

## Project Structure

```
MCP Project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
├── mcp_client_final.py           # Main application
├── create_index.py               # Knowledge base indexing script
├── run_mcp_server.py             # MCP server launcher (optional)
├── IT_Support_Knowledge_Base.pdf # Source PDF document
├── kb_faiss_index.bin            # Generated FAISS index
└── kb_chunks.pkl                 # Generated chunks metadata
```

## Technical Details

### Knowledge Base Indexing (`create_index.py`)

- **PDF Extraction**: Uses `pdfplumber` to extract text
- **Chunking Strategy**: Splits by numbered section headings (e.g., "1. Title", "2. Title")
- **Embeddings**: Uses `all-MiniLM-L6-v2` sentence transformer model (384 dimensions)
- **Index Type**: FAISS IndexFlatL2 for exact L2 distance search

### Main Application (`mcp_client_final.py`)

- **LLM**: OpenAI GPT models via LangChain
- **RAG**: FAISS vector search + sentence-transformers
- **MCP Integration**: Uses `mcp-atlassian` for Jira operations
- **Tool Orchestration**: LangChain tool binding for function calling

## Troubleshooting

### Knowledge Base Not Found

If you see warnings about missing KB files:
- Run `create_index.py` first to generate the index
- Ensure `IT_Support_Knowledge_Base.pdf` exists in the project directory

### MCP Server Connection Issues

- Verify your `.env` file has correct Jira credentials
- Check that `mcp-atlassian` is installed: `pip install mcp-atlassian`
- Ensure your Jira API token is valid

### OpenAI API Errors

- Verify `OPENAI_API_KEY` is set in `.env`
- Check your OpenAI account has sufficient credits
- Ensure the API key has proper permissions

## Dependencies

See `requirements.txt` for the complete list. Key packages:

- `mcp-atlassian` - Jira MCP server
- `langchain-openai` - OpenAI integration
- `pdfplumber` - PDF processing
- `faiss-cpu` - Vector search
- `sentence-transformers` - Text embeddings
- `python-dotenv` - Environment management

## License

This project is provided as-is for IT support automation purposes.

## Support

For issues or questions:
1. Check that all prerequisites are installed
2. Verify `.env` configuration
3. Ensure the knowledge base index has been created
4. Review error messages for specific guidance

