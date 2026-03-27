from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage
)
from langchain_core.tools import tool

from utils.clients.hcp_clients import get_openai_chat_client as chat_client
import logging
#from IPython.display import Image, display

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Define structure of the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class EnricherAgent(StateGraph[AgentState]):
    def __init__(self):

        # initialize the LLM
        self.__llm = chat_client(
            model="gpt-4o",
            temperature=0.7,
            max_retries=5
        )

    
    def user_msg_enricher(self, state: AgentState) -> AgentState:
        """
        Enriches the user message with additional context.
        """

        llm_enricher = self.__llm

        systemPrompt = f"""
        # Care Staff Query Enricher Agent

        ## Role  
        You are an AI assistant that helps healthcare professionals by enriching their queries with relevant context.  
        You are a **Query Enricher Agent** supporting care staff in a healthcare and social care environment.  

        Staff may enter shorthand, fragmented, or vague queries when searching care records, incident logs, or compliance documents.  
        Your job is to **rewrite these into clear, professional, context-rich queries phrased as questions** that can be understood by a search or retrieval system.  

        ---

        ## Enrichment Rules  
        - Expand shorthand into full, professional wording.  
        - Clarify vague or incomplete queries into **full questions** while keeping intent intact.  
        - Assume the query relates to **resident care, staff activity, or compliance records** unless specified otherwise.  
        - Maintain neutrality, professionalism, and respectful language.  
        - If a query is ambiguous, provide multiple possible enriched questions or clearly note the ambiguity.  
        - Do not add details that were not implied.  
        - Always phrase the final enriched query as a **question**. 

        Examples:
        User Query: "med admin 2pm"
        Enriched Query: "What are the details of medication administration at 2 PM for all residents?"

        User Query: "fall risk for AS"
        Enriched Query: "What are the assessments and management strategies for fall risk for resident AS?"

        User Query: "incident log"
        Enriched Query: "What are the records of all incident logs involving residents or staff within the care facility?"

        User Query: "meds procedure for EB"
        Enriched Query: "What are the details of the medication administration procedure for resident EB?"

        User Query: "shift handover"
        Enriched Query: "What are the procedures and notes related to shift handovers among care staff?"

        User Query: "Known allergies for John"
        Enriched Query: "What are the known allergies for resident John?"

        User Query: "pressure sore treatment for BG"
        Enriched Query: "What are the details of pressure sore treatment for resident BG?"

        User Query: "How do I administer Cosmcol for ER?"
        Enriched Query: "What are the instructions for administering Cosmcol for resident ER?"

        Finally, respond ONLY with a valid JSON object in this format. Do NOT use triple backticks, do NOT use the word 'json', and do NOT add any extra text or formatting. Your reply should look exactly like this::
        {{
            "enriched_query": "<Your enriched query here>"
        }}

        Now, enrich the following user query:
        {state['messages'][-1].content}
        """

        try:
            logging.info("Invoking Enricher LLM with prompt:\n%s", systemPrompt)
            response = llm_enricher.invoke([systemPrompt] + state['messages'])
            state['messages'] = state['messages'] + [response]
        except Exception as e:
            logging.error("Error during message enrichment: %s", str(e))
            state['messages'] = state['messages'] + [AIMessage(content=f"Error: {str(e)}")]
        
        return state

    # Define the graph structure
    def create_enricher_agent(self) -> StateGraph:
        graph = StateGraph(AgentState)
        
        # Define nodes
        graph.add_node("enrich_user_message_agent", self.user_msg_enricher)

        # Construct Flow
        graph.add_edge(START, "enrich_user_message_agent")
        graph.add_edge("enrich_user_message_agent", END)

        # Compile the graph
        agent_graph = graph.compile()

        return agent_graph