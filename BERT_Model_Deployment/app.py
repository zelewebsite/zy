import numpy as np
from flask import Flask, render_template, request
import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn


app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer for Afaan Oromoo fake news detection

#Load BERT model and tokenizer using Huggingface transformers
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#freeze bert layers
#freezing the parameters and defining trainable BERT structure
for param in bert.parameters():
    param.requires_grad = False # no gradient computation here

#Define Our Model Architecture

class BERT_Arch(nn.Module):

  def __init__(self, bert):
    super(BERT_Arch, self).__init__()
    self.bert = bert
    self.dropout = nn.Dropout(0.1)  #dropout layer
    self.relu = nn.ReLU() # activation function
    self.fc1 = nn.Linear(768,512) # first dense layer
    self.fc2 = nn.Linear(512,2)  # second dense layer (output layer)
    self.softmax = nn.LogSoftmax(dim=1) #activation function
  def forward(self, sent_id, mask):#forward pass function definition,
    to_hs_before = self.bert(sent_id, attention_mask=mask)["last_hidden_state"]
    to_hs = to_hs_before  # Access the hidden states tensor
    # Check if to_hs is not NoneType
    if to_hs is not None:
        to_hs = torch.tensor(to_hs, dtype=torch.float32)  # Convert to torch.Tensor

    # Check the shape of to_hs.hidden_states
    if to_hs.shape[2] != 768:
        raise ValueError("Invalid input shape for fc1 layer.")

    # Check the data type of to_hs.hidden_states
    if not torch.is_floating_point(to_hs):
        to_hs = to_hs.float()

    # Check if to_hs is on the same device as the model
    if to_hs.device != self.fc1.weight.device:
        to_hs = to_hs.to(self.fc1.weight.device)

    x = self.fc1(to_hs)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.softmax(x)

    return x


model = BERT_Arch(bert)

model_path = 'models/afaan_oromo_fake_news_pred_model.pt' # Model path


model.load_state_dict(torch.load(model_path))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_data = [request.form['news']]
  
    MAX_LENGTH = 15
    
    tokens_unseen = tokenizer.batch_encode_plus(news_data,
                                          max_length = MAX_LENGTH,
                                          pad_to_max_length = True,
                                          truncation = True)
    unseen_seq  = torch.tensor(tokens_unseen['input_ids'])
    unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

    with torch.no_grad():
         preds = model(unseen_seq, unseen_mask)
    predicted_class = np.argmax(preds, axis = 1)

    pred_class = predicted_class.reshape(-1)

    result = pred_class[0]
    
    if result == 1:
        return render_template('index.html', prediction_text = 'The given news article has been classified as FAKE!\n Odeeffannoon kun DOGOGGORA!')
    else:
        return render_template('index.html', prediction_text = 'The given news article has been classified as REAL!\n Odeeffannoon kun SIRRIIDHA!')
        
if __name__ == "__main__":
    app.run()
