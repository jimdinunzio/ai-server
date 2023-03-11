from transformers import GPTJForCausalLM, AutoTokenizer
import torch

class GptJChatbot:
    def __init__(self, mname, init_prompt, intro_line, name1_label, name2_label):
        self.mname = mname
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = GPTJForCausalLM.from_pretrained(mname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(mname)
        self.init_prompt = init_prompt
        self.chat_log = ''
        self.reply_ids = None
        self.name1_label = name1_label
        self.name2_label = name2_label
        self.intro_line = intro_line

    def get_intro_line(self):
        return self.intro_line
    
    def get_response(self, input):
        prompt = f'{self.init_prompt}{self.chat_log}\n{self.name1_label}: {input}\n{self.name2_label}: '
        #print(f'submitting:\n{prompt}\n-------------------------------\n')

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        self.reply_ids = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=1.0,
            temperature=0.9,
            max_new_tokens=50,
            min_length=8,
            repetition_penalty=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        ).to(self.device)
         
        full_resp = self.tokenizer.batch_decode(self.reply_ids)[0]
        #full_resp = f'{prompt}Thats very interesting. Tell me more.\nHuman:'
        resp = full_resp.replace(f"{self.init_prompt}{self.chat_log}", "")
        print(f"\n------------------------------\n{resp}\n-----------------------------\n")
        re = resp.split(f"\n{self.name2_label}:")           # split at robot: 
        if self.name1_label in re:                          # if Human: in reply then 
            res = re[1].split(f"\n{self.name1_label}:")[0]  # take the text between robot: and human:
        else:                                               # otherwise split at the next newline
            res = re[1].split("\n")[0]
        if "." in res:
            res = res.strip().rpartition('.')[0]            # strip white space and take up to last period.
            if len(res):
                res += '.'
        else:
            res = res.strip()
    
        return res
    
    def add_to_chat_log(self, name1_utt, name2_utt):
        self.chat_log += f'\n{self.name1_label}:{name1_utt}\n{self.name2_label}:{name2_utt}'

    def init_chat_log(self):
        self.chat_log = ''

    def get_log(self):
        return self.chat_log
    