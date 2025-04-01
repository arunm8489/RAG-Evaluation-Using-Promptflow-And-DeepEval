from deepeval.models.base_model import DeepEvalBaseLLM
from azure.identity import ClientSecretCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
import os
load_dotenv()



def get_model():
    token_provider = get_bearer_token_provider(
                    ClientSecretCredential(os.getenv('AZURE_OPENAI_TENANT_ID'), os.getenv('AZURE_OPENAI_CLIENT_ID'), os.getenv('AZURE_OPENAI_CLIENT_SECRET')), 
                    os.getenv('AZURE_OPENAI_TOKEN_BASE')
                )

    # Create an instance of the AzureOpenAI class
    aoai = AzureOpenAI(
                    api_version="2024-08-01-preview",
                    azure_endpoint= os.getenv('AZURE_OPENAI_API_BASE'),
                    azure_ad_token_provider= token_provider
                )

    return aoai




class EvalAzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        messages = [{"content": prompt, "role": "system"}]
        return chat_model.chat.completions.create(
            messages=messages,
            model='gpt-4o',
            max_tokens=3000,
            temperature=0.0,
            seed = 123,
        ).choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Azure OpenAI AAD Eval Model"