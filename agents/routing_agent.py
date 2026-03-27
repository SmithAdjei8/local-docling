from typing import TypedDict, Annotated, Sequence, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from utils.clients.hcp_clients import get_openai_chat_client as chat_client

import logging
# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Define structure of the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    selected_agent: Optional[str]

# RoutingAgent definition
class RoutingAgent(StateGraph[AgentState]):

    def __init__(self, agent_details: List[dict]):

        self.agent_details = agent_details

        # Initialize LLM
        self.__llm = chat_client(
            model="gpt-4o",
            temperature=0.0,
            max_retries=5
        )

    def route_query(self, state: AgentState) -> AgentState:
        """
        Routes the enriched query to the appropriate specialist agent.
        """
        system_prompt = f"""
        You are a healthcare query routing agent.
        Your role is to analyze the user’s query and determine which specialist agent(s) should handle it. 
        You must decide based on the available agents’ expertise. 

        Rules:
        1. If the query clearly belongs to a single specialist agent’s domain, return that agent’s name.
        2. If the query requires collaboration between multiple specialist agents, return ALL relevant agent names as a comma-separated list in the correct execution order.
        - For example: if a query first requires retrieving data (Incident Log Agent) and then checking rules (Compliance Document Agent), return: "Incident Log Agent, Compliance Document Agent".
        3. If the query is not healthcare-related or does not match any agent, return "None".
        4. Do not include any explanation — return only the agent name(s).

        The available specialist agents are:
        {self.agent_details}

        Example:
        User Query: "Show me Mrs. Smith’s last incident report and check if it matches compliance rules."
        Expected Output: "Incident Log Agent, Compliance Document Agent"
        User Query: "What are the known allergies for resident John?"
        Expected Output: "Care Record Agent"
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Enriched Query: {state['messages'][-1].content}\n\n"
                        "Which agent is best suited to handle this query? Respond with the agent name only, or 'None'."
            )
        ]

        try:
            response = self.__llm.invoke(messages)
            selected_agent = response.content.strip()
            logging.info(f"Routing Agent selected: {selected_agent}")
            state['selected_agent'] = selected_agent
        except Exception as e:
            logging.error(f"Error during routing: {e}")
            state['selected_agent'] = "None"

        return state

    def create_routing_graph(self, agent_details: List[dict]) -> StateGraph:
        """
        Creates the LangGraph nodes and edges for routing.
        """
        graph = StateGraph(AgentState)
        graph.add_node("route_query", self.route_query)
        graph.set_entry_point("route_query")
        graph.add_edge("route_query", END)

        agent_graph = graph.compile()

        
        logging.info("Routing graph created and compiled.")
        return agent_graph
    
# if __name__ == "__main__":
#     # Example usage
#     agent_details = [
#         {"name": "Care Record Agent", "expertise": "Accessing and managing resident care records."},
#         {"name": "Incident Log Agent", "expertise": "Retrieving and analyzing incident logs."},
#         {"name": "Compliance Document Agent", "expertise": "Checking compliance documents and regulations."}
#     ]

#     routing_agent = RoutingAgent(agent_details)
#     routing_graph = routing_agent.create_routing_graph(agent_details)

#     initial_state: AgentState = {
#         "messages": [HumanMessage(content="What are the known allergies for resident John?")],
#         "selected_agent": None
#     }

#     final_state = routing_graph.invoke(initial_state)
#     print(final_state)