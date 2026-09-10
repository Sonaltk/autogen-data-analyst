import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

async def main():
    model_client = OllamaChatCompletionClient(model="qwen2.5:7b")

    agent = AssistantAgent(
        name="test_agent",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )

    result = await agent.run(task="Say hello and tell me you're running locally via Ollama.")
    print(result.messages[-1].content)

asyncio.run(main())
