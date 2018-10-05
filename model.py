import torch
import torch.nn as nn
import torch.optim as optim

from torchcrf import CRF

torch.manual_seed(1)

CUDA = True if torch.cuda.is_available() else False

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

# Create model
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx = 1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.emissons = nn.Linear(hidden_dim, self.tagset_size)

        self.hidden = self.init_hidden()
        
        self.crf = CRF(len(self.tag_to_ix))

    def init_hidden(self):
        return (randn(2, self.batch_size, self.hidden_dim // 2),
                randn(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        """
        sentence : (sent, batch)
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.emissons(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags):
        """
        sentence : (sent, batch)
        tags : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        # Computing log likelihood
        mask = sentence.ne(1)  # (s, b)
        llh = self.crf(emissions, tags, mask=mask)
        return - llh
    
    def predict(self, sentence):
        """
        sentence : (sent, batch)
        """
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask)