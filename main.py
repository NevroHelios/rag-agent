import os

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from colorama import Fore
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
    
load_dotenv()

model = LitellmModel(
        model="groq/llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )



set_tracing_disabled(True)

@function_tool
def get_weather_info(place: str) -> str:
    """
    This function returns the weather information for a given place.
    Args:
        place (str): The name of the place to get weather information for.
    Returns:
        str: The weather information for the given place.
    """
    engine = DuckDuckGoSearchResults()
    search_results = engine.run(f"How is the weather in {place}")
    weather_info = search_results if search_results else "No weather information found."
    return weather_info

agent = Agent(
    name="Weather agent",
    instructions="You are a weather agent. Your task is to provide weather information. "
                 "Use the 'get_weather_info' tool to find the weather for the requested place. "
                 "Only use the tools provided to you. Do not use any other tools.",
                 
    model=model,
    tools=[get_weather_info],
    model_settings={
        "temperature": 0.1
    }
)


async def main():
    result = await Runner.run(agent, "What is the weather in kolkata?")
    print(result)
    print(Fore.YELLOW + result.final_output + Fore.RESET)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())