import re
import random

with open('/root/Desktop/factor_mining/fig/domian_factors_prompts/A_initial_expressions.txt', 'r', encoding='utf-8') as f:
    content = f.read()

def replacer(match):
    return str(random.choice([10, 20, 30, 50]))

new_content = re.sub(r'%d', replacer, content)

with open('/root/Desktop/factor_mining/fig/domian_factors_prompts/A_initial_expressions.txt', 'w', encoding='utf-8') as f:
    f.write(new_content)
