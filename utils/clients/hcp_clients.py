import os
from langchain_openai import (
    AzureChatOpenAI, 
    AzureOpenAIEmbeddings,
    ChatOpenAI, 
    OpenAIEmbeddings
)

from functools import cache
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

@cache
def get_cosmos_client() -> CosmosClient:
     endpoint = os.getenv("COSMOS_DB_ENDPOINT"),
     credential = DefaultAzureCredential()
     return CosmosClient(endpoint, credential)
     

@cache
def get_azure_chat_client(deployment_name: str,
                          temperature: float = 0.7,
                          max_retries: int = 5,
                          streaming: bool = False
                          ) -> AzureChatOpenAI:
        config_opts: dict = {
             "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
             "azure_deployment": deployment_name,
             "model": deployment_name,
             "max_retries": max_retries,
             "streaming": streaming,
        }
        if temperature is not None:
             config_opts["temperature"] = temperature

        return AzureChatOpenAI(
             **config_opts
        )

@cache
def get_azure_embeddings_client(deployment: str) -> AzureOpenAIEmbeddings:
     return AzureOpenAIEmbeddings(
          azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
          azure_deployment=deployment,
     )

@cache
def get_openai_chat_client(model: str, 
                    temperature: float = 0.7, 
                    max_retries: int = 3) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
        temperature=temperature,
        max_retries=max_retries
    )

@cache
def get_openai_embeddings_client(model:str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )