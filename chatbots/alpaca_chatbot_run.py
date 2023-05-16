from alpaca_chatbot import AlpacaChatbot

class AlpacaChatbotRun:
    def __init__(self, size, load_8bit):
        self.chat_bot = None
        self.size = size
        self.load_8bit = load_8bit

    def start(self):    
        prompt_name ="helpful_assistant_prompt"
        init_prompt = ""
        with open(f'prompts/{prompt_name}.txt', 'r') as f:
            init_prompt = f.read()
        
        intro_line = "I am a helpful assistant. Say bye to end the chat. What can I answer for you today?"
        self.chat_bot = AlpacaChatbot("decapoda-research/llama-"+self.size+"-hf", "training/alpaca-lora/lora-alpaca-"+self.size,
                                      init_prompt, intro_line, "Human", "AI", load_8bit=self.load_8bit)
        return self.chat_bot
        
    def stop(self):
        del self.chat_bot
        self.chat_bot = None

if __name__ == "__main__":
    bot = AlpacaChatbotRun(size="7b", load_8bit=False)
    chat_bot = bot.start()
    inp = ""
    print(chat_bot.intro_line)
    while inp != "bye":
        print("> ",end='')
        inp = input()
        if inp == ".log":
            print(chat_bot.get_log())
            continue
        if len(inp) == 0:
            continue
        short, answer, long = chat_bot.get_response(inp)
        #print(f"\n------------------------------\n{long}\n-----------------------------")
        chat_bot.add_to_chat_log(inp, answer)
        print(answer)
    bot.stop()

    