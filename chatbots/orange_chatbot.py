from gpt_j_chatbot import GptJChatbot

class OrangeChatbot:
    def __init__(self):
        self.chat_bot = None

    def start(self):
        prompt_name ="orange_prompt"
        init_prompt = ""
        with open(f'prompts/{prompt_name}.txt', 'r') as f:
            init_prompt = f.read()
        
        intro_line = "In chat mode I can answer questions about myself. Say goodbye to end the chat. What can I answer for you today?"
        self.chat_bot = GptJChatbot("gpt-j-6b", init_prompt, intro_line, "Human", "Robot")
        return self.chat_bot
        
    def stop(self):
        del self.chat_bot
        self.chat_bot = None

if __name__ == "__main__":
    bot = OrangeChatbot()
    chat_bot = bot.start()
    inp = ""
    print(chat_bot.intro_line)
    while inp != "bye":
        print("> ",end='')
        inp = input()
        if inp == ".log":
            print(chat_bot.get_log())
            continue
        answer, _, long = chat_bot.get_response(inp)
        print(f"\n------------------------------\n{long}\n-----------------------------\n")
        chat_bot.add_to_chat_log(inp, answer)
        print(answer)
    bot.stop()

    