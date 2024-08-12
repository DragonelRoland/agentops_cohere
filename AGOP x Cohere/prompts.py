import os
from cohere import Client as CohereClient

# Initialize the Cohere client
cohere_api_key = os.getenv('COHERE_API_KEY')
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable is not set")
client = CohereClient(api_key=cohere_api_key)

PROMPT_TYPES = {
    'ui_vector': "Generate a research vector based on the following context: {context}",
    'scraper_analysis': "Analyze the following search results in the context of: {context}",
    'analysis_options': "Provide analysis options for the following data: {data}",
    'analysis_key_info': "Extract key information from the following data: {data}",
    'analysis_presentation_options': "Suggest presentation options for the following key information: {key_info}",
    'output_generation': "Generate a {output_style} output for the following key information: {key_info}",
    'final_feedback': "Provide feedback on the following final document: {document}"
}

def generate_prompt(context: str, prompt_type: str) -> str:
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    prompt_template = PROMPT_TYPES[prompt_type]
    return prompt_template.format(context=context)

# Add these lines at the end of the file
if __name__ == "__main__":
    print("COHERE_API_KEY:", "Set" if os.getenv('COHERE_API_KEY') else "Not set")
    print("Cohere client initialized:", client is not None)
    print("Available prompt types:", list(PROMPT_TYPES.keys()))
    print("Sample prompt:", generate_prompt("test context", "ui_vector"))