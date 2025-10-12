from llm.openrouter import OpenRouterLLM
from llm.base import LLMPromptTemplate

def main():
    test_model = OpenRouterLLM(
        model_name="llama_small"
        )
    prompt = "What is the capital of France?"
    response = test_model.prompt_model(
        prompt=LLMPromptTemplate(user_prompt=prompt)
        )
    print(test_model.parse_response(response))
    

if __name__ == "__main__":
    main()