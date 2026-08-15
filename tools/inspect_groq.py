import inspect
from groq import Groq

client = Groq()
fn = client.chat.completions.create
print("Signature:", inspect.signature(fn))
print("\nDocstring:\n", fn.__doc__)
