import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ProactiveChatbot:
    def __init__(self, model_path):
        # Initialize the DialoGPT tokenizer and model from the specified local directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.agents = []

    def add_agent(self, role, prompt):
        self.agents.append({"role": role, "prompt": prompt})

    def reset(self):
        self.agents = []

    def generate_bot_response(self, user_input, conversation_history):
        if not conversation_history:
            # Bot initiates conversation
            return self.agents[0]["prompt"]
        elif conversation_history and len(conversation_history) < len(self.agents):
            # Continue with the next agent's prompt
            return self.agents[len(conversation_history)]["prompt"]
        else:
            # Handle small talk or user questions using the DialoGPT model
            inputs = self.tokenizer.encode(user_input, return_tensors='pt')
            bot_input_ids = torch.cat([conversation_history, inputs], dim=-1) if len(conversation_history) > 0 else inputs
            bot_response = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1)
            return self.tokenizer.decode(bot_response[0], skip_special_tokens=True)

def main():
    st.title("Proactive Chatbot with DialoGPT-large")
    model_path = "/home/abhishek/projects/llm-chatbot-for-information-extraction/DialoGPT-large"
    chatbot = ProactiveChatbot(model_path)

    # Add agents with roles and prompts
    chatbot.add_agent("convincing", "I'm here to help you. Can you share some details with me?")
    chatbot.add_agent("details", "Great! Please provide your name.")
    chatbot.add_agent("details", "Thank you! Now, please provide your email address.")
    chatbot.add_agent("details", "Excellent! What's your phone number?")
    chatbot.add_agent("details", "Fantastic! Can you share your address, including street, city, and country?")
    chatbot.add_agent("details", "Wonderful! When were you born? Please provide your date of birth.")
    chatbot.add_agent("details", "Thank you! Could you tell me about your education? Where did you study and what is your highest qualification?")
    chatbot.add_agent("smalltalk", "I understand that sharing information can be a sensitive topic. Let's talk about something else for a moment. What's your favorite hobby or interest.")

    # Conversation history
    conversation_history = []

    # Button to send user input
    user_input = st.text_input("You: Enter your message", "")

    if st.button("Send"):
        if user_input:
            # Append user's message to the conversation history
            conversation_history.append(user_input)
            
            # Get the bot's response
            bot_response = chatbot.generate_bot_response(user_input, conversation_history)
            st.write(f"Bot: {bot_response}")

    # Reset the conversation
    if st.button("Reset Conversation"):
        conversation_history.clear()
        chatbot.reset()
        st.write("Bot: Conversation has been reset.")

if __name__ == "__main__":
    main()
