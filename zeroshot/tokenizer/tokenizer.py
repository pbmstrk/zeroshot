from transformers import AutoTokenizer

class ZeroShotTopicTokenizer:

    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, text, **kwargs):

        if isinstance(text, str):
            text = [text]
        if isinstance(text, tuple):
            text = list(text)

        encoded_text = self.tokenizer(text, **kwargs)
        return encoded_text
