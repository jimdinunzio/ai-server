from transformers import pipeline, Conversation
import torch

class BlenderbotConversation:
    def __init__(self, mname, prompt):
        self.mname = mname
        self.device=0 if torch.cuda.is_available() else -1
        self.converse = pipeline("conversational", model=self.mname, device=self.device)
        self.prompt = prompt
        self.chat = None
        
    def start(self):
        self.chat = Conversation(self.prompt)
        self.chat = self.converse(self.chat)

    def get_response(self, input=None):
        if input is not None:
            self.chat.add_user_input(input)    
            self.chat = self.converse(self.chat)
        return self.chat.generated_responses[-1]


from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

class BlenderbotManualConversation:
    def __init__(self, mname, init_prompt):
        self.mname = mname
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BlenderbotForConditionalGeneration.from_pretrained(mname).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(mname)
        self.prompt = init_prompt
        self.reply_ids = None
        
    def get_response(self, input):
        self.prompt += " Human: " + input
        print("submitting: ", self.prompt)
        inputs = self.tokenizer([self.prompt], return_tensors="pt").to(self.device)
        self.reply_ids = self.model.generate(**inputs).to(self.device)
        resp = self.tokenizer.batch_decode(self.reply_ids, skip_special_tokens=True)[0]
        self.prompt += " Orange: " + resp
        return resp
    

if __name__ == "__main__":
    init_prompt = "The following conversation is between a robot named Orange and a human. " \
             "Orange is a helpful personal assistant robot built in 2020 by Jim DiNunzio. " \
             "It has capabilities of indoor navigation, object and person identification. " \
             "It has an AI depth camera, a LiDAR, bump sensors and radar. Orange: Hi, I am Orange. How can I help you today?"
    chat_bot = BlenderbotManualConversation("blenderbot-3b", init_prompt)
    inp = ""
    print("Hi, I am Orange. How can I help you today?")
    while inp != "bye":
        print("> ",end='')
        inp = input()
        resp = chat_bot.get_response(inp)
        print(resp)
    