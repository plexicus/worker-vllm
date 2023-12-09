class Template():
    def __init__(self, template_method):
        self.template_method = template_method

    def __call__(self, prompt):
        return self.template_method(prompt)


LLAMA2_TEMPLATE = Template(
    lambda prompt: """SYSTEM: You are a helpful assistant.
USER: {}
ASSISTANT: """.format(prompt)
)

DEFAULT_TEMPLATE = Template(
    lambda prompt: prompt
)

DEEPSEEK_TEMPLATE = Template(
    lambda prompt: f'''You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer

### Instruction:

{prompt}

### Response: '''
)