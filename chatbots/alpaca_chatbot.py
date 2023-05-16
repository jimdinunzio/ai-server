import sys
from peft import PeftModel
import transformers
import torch

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

class AlpacaChatbot:
    def __init__(self, base_model, lora_weights, init_prompt, intro_line, name1_label, name2_label, load_8bit):
        self.LOAD_8BIT = load_8bit
        self.BASE_MODEL = base_model
        self.LORA_WEIGHTS = lora_weights
        self.init_prompt = init_prompt
        self.tokenizer = LlamaTokenizer.from_pretrained(self.BASE_MODEL)
        self.device="cuda" if torch.cuda.is_available() else "cpu"

        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:
            pass

        if self.device == "cuda":
            self.model = LlamaForCausalLM.from_pretrained(
                self.BASE_MODEL,
                load_in_8bit=self.LOAD_8BIT,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                self.LORA_WEIGHTS,
                torch_dtype=torch.float16,
                device_map={'': 0},
            )
        elif self.device == "mps":
            self.model = LlamaForCausalLM.from_pretrained(
                self.BASE_MODEL,
                device_map={"": self.device},
                torch_dtype=torch.float16,
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                self.LORA_WEIGHTS,
                device_map={"": self.device},
                torch_dtype=torch.float16,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.BASE_MODEL, device_map={"": self.device}, low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                self.LORA_WEIGHTS,
                device_map={"": self.device},
            )

        # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not self.LOAD_8BIT:
            self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(model)

        self.chat_log = ''
        self.reply_ids = None
        self.name1_label = name1_label
        self.name2_label = name2_label
        self.intro_line = intro_line

    def generate_prompt(self, instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""
        
    def evaluate(
        self,
        instruction,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        #prompt = self.generate_prompt(instruction, self.chat_log)
        prompt = f'{self.init_prompt}{self.chat_log}\n{self.name1_label}: {instruction}\n{self.name2_label}: '
        #print(f'\nSUBMITTING:\n{prompt}\n-------------------------------\n')
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return output

    def get_intro_line(self):
        return self.intro_line
    
    def get_response(self, input):
        full_resp = self.evaluate(input)
        long_resp = full_resp.replace(f"{self.init_prompt}{self.chat_log}", "")
        long_resp = long_resp.strip()
        #print(f"--------------------------\n{long_resp}\n-------------------------------")
        # split at <name2_label>: (chatbot)
        med_resp = long_resp.split(f"\n{self.name2_label}:")
        # if <name2_label>: (input source) in reply then take the text between it and <name1_label>: (input source)
        if len(med_resp) > 0:
            if self.name1_label+':' in med_resp[1]:
                med_resp = med_resp[1].split(f"\n{self.name1_label}:")[0]
                short_resp = med_resp
            elif self.name2_label+':' in med_resp[1]:
                med_resp = med_resp[1].split(f"\n{self.name2_label}:")[0]
                short_resp = med_resp
            else:                                          
                # otherwise split at the next newline for short response, to end for medium resp
                med_resp = med_resp[1]
                short_resp = med_resp[1].split("\n")[0]
        else:
            med_resp = med_resp[0]
            short_resp = med_resp
        # strip white space and take up to last period for short resp.
        med_resp = med_resp.strip()
        if "." in short_resp:
            short_resp = short_resp.strip().rpartition('.')[0]
            if len(short_resp):
                short_resp += '.'
        else:
            short_resp = short_resp.strip()

        #print(f"----------------------------------\nSHORT = {short_resp}\n----------------------\nMED = {med_resp}\n----------------------------------\nLONG = {long_resp}\n---------------------------")
        return short_resp, med_resp, long_resp

    def add_to_chat_log(self, name1_utt, name2_utt):
        # if '\n' in name2_utt:
        #     name2_utt = "    " + name2_utt.replace('\n', '\n    ')
        self.chat_log += f'\n{self.name1_label}: {name1_utt}\n{self.name2_label}: {name2_utt}\n'

    def init_chat_log(self):
        self.chat_log = ''

    def get_log(self):
        return self.chat_log
    