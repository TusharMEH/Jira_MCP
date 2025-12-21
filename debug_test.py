"""
Quick debug test to see actual MCP server errors
"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()


async def main():
    print("Starting debug test...")
    print(f"Jira URL: {os.getenv('JIRA_URL')}")
    
    server_params = StdioServerParameters(
        command="mcp-atlassian",
        args=["--env-file", ".env", "-vv"],
        env={**os.environ}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected\n")
            
            # Test 1: List all projects first
            print("=" * 50)
            print("TEST 1: List all projects")
            print("=" * 50)
            try:
                result = await session.call_tool("jira_get_all_projects", {})
                for item in result.content:
                    if hasattr(item, 'text'):
                        print(item.text[:1000])
            except Exception as e:
                print(f"Error: {e}")
            
            # Test 2: Try to create an issue
            print("\n" + "=" * 50)
            print("TEST 2: Create issue in IT project")
            print("=" * 50)
            try:
                result = await session.call_tool("jira_create_issue", {
                    "project_key": "IT",
                    "summary": "Test issue from MCP",
                    "issue_type": "Task",
                    "description": "This is a test issue"
                })
                for item in result.content:
                    if hasattr(item, 'text'):
                        print(f"Result: {item.text}")
            except Exception as e:
                print(f"Error: {e}")
            
            # Test 3: Try with Bug type
            print("\n" + "=" * 50)
            print("TEST 3: Create Bug in IT project")
            print("=" * 50)
            try:
                result = await session.call_tool("jira_create_issue", {
                    "project_key": "IT",
                    "summary": "Test bug from MCP",
                    "issue_type": "Bug",
                    "description": "This is a test bug"
                })
                for item in result.content:
                    if hasattr(item, 'text'):
                        print(f"Result: {item.text}")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())