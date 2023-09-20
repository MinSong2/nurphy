import torch
from transformers import ElectraModel, ElectraTokenizer
from scipy.spatial.distance import cosine

device = torch.device("cpu")

text = "예로부터 말이 많은 사람은 말로 망하고 밤중에 말 타고 토끼를 데리고 도망치는 경우가 많다."
marked_text = "[CLS] " + text + " [SEP]"

model = ElectraModel.from_pretrained("monologg/koelectra-base-v2-discriminator",
                                     output_hidden_states = True)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v2-discriminator")

model.to(device)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")
# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

print(list(tokenizer.vocab.keys())[5000:5020])

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))


#segment ID
# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

print (segments_ids)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers.
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)

    print(str(len(outputs)))
    # Evaluating the model will return a different number of objects based on
    # how it's  configured in the `from_pretrained` call earlier. In this case,
    # becase we set `output_hidden_states = True`, the third item will be the
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[1]


print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

# For the 5th token in our sentence, select its feature values from layer 5.
token_i = 5
layer_i = 5
vec = hidden_states[layer_i][batch_i][token_i]

# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())

#Let’s combine the layers to make this one whole big tensor.
# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(hidden_states, dim=0)
print(str(token_embeddings.size()))

#Let’s get rid of the “batches” dimension since we don’t need it.
# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)
print(str(token_embeddings.size()))

#Finally, we can switch around the “layers” and “tokens” dimensions with permute.
# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)
print(str(token_embeddings.size()))


# Stores the token vectors, with shape [22 x 3,072]
token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    # `token` is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)

print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))


#As an alternative method, let’s try creating the word vectors by summing together the last four layers.
# Stores the token vectors, with shape [22 x 768]
token_vecs_sum = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
    sum_vec = torch.sum(token[-4:], dim=0)

    # Use `sum_vec` to represent `token`.
    token_vecs_sum.append(sum_vec)

print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


# `token_vecs` is a tensor with shape [22 x 768]
token_vecs = hidden_states[-2][0]

# Calculate the average of all 22 token vectors.
sentence_embedding = torch.mean(token_vecs, dim=0)
print ("Our final sentence embedding vector of shape:", sentence_embedding.size())


for i, token_str in enumerate(tokenized_text):
  print (i, token_str)

'''
They are at 3, 9, and 15.

For this analysis, we’ll use the word vectors that we created by summing the last four layers.

We can try printing out their vectors to compare them.
'''
print('First 10 vector values for each instance of "말".')
print('')
print("말 (언어) ", str(token_vecs_sum[3][:10]))
print("말 (언어)  ", str(token_vecs_sum[9][:10]))
print("말 (동물)   ", str(token_vecs_sum[15][:10]))

'''
We can see that the values differ, but let’s calculate the cosine similarity between the vectors to make a more precise comparison.
'''

# Calculate the cosine similarity between the word
diff_word = 1 - cosine(token_vecs_sum[3], token_vecs_sum[15])

# Calculate the cosine similarity between the word
same_word = 1 - cosine(token_vecs_sum[3], token_vecs_sum[9])

print('Vector similarity for  *similar*  meanings:  %.2f' % same_word)
print('Vector similarity for *different* meanings:  %.2f' % diff_word)