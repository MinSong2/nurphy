from transformers import pipeline, set_seed

set_seed(42)

#language generation
generator = pipeline('text-generation', model='gpt2')
results = generator("Hello, I'm a language model,")
print(results)

#classification
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
#classifier = pipeline('zero-shot-classification', model='gpt2')
text_piece = "The food at this place is really good."
labels = ["Food", "Employee", "Restaurant", "Party", "Nature", "Car"]
predictions = classifier(text_piece, labels, multi_class=False)
print(predictions)

#summarization
summarizer = pipeline('summarization')
text = "He is one of only eight people to be granted honorary citizenship of the United States; others include Lafayette, Raoul Wallenberg and Mother Teresa.[463] The United States Navy honoured him in 1999 by naming a new Arleigh Burke-class destroyer as the USS Winston S. Crchill.[464] Other memorials in North America include the National Churchill Museum in Fulton, Missouri, where he made the 1946 \"Iron Curtain\" speech; Churchill Square in central Edmonton, Alberta; and the Winston Churchill Range, a mountain range northwest of Lake Louise, also in Alberta, which was renamed after Churchill in 1956."
summary=summarizer(text, min_length=5, max_length=30)
print(summary)

#translation - 1
translator = pipeline("translation_en_to_de") #translation_en_to_ko
text = "Hello my friends! How are you doing today?"
translation = translator(text)
print(translation)

#translation - 2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

text = "Hello my friends! How are you doing today?"
tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')

translation = model.generate(**tokenized_text)
translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

print(translated_text)