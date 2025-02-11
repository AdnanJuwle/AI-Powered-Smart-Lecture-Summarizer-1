import numpy as np
import random
questions = [
    "What's your favorite programming language?",
    "Name a place you'd love to visit.",
    "What's your dream job?",
    "What's your favorite hobby?"
]
question = random.choice(questions)
answer = input(question + " ")
print(f"Wow! No way you like {answer}. You are dumb!")
