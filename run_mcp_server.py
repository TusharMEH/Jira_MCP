import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify credentials are loaded
print("Checking credentials...")
print(f"JIRA_URL: {os.getenv('JIRA_URL')}")
print(f"JIRA_USERNAME: {os.getenv('JIRA_USERNAME')}")
print(f"JIRA_API_TOKEN: {'***' + os.getenv('JIRA_API_TOKEN', '')[-4:] if os.getenv('JIRA_API_TOKEN') else 'NOT SET'}")

# Run the MCP server using the same Python interpreter
print("\nStarting MCP Atlassian server...")
os.system(f'"{sys.executable}" -m mcp_atlassian')