
import torch.nn as torch_nn
import librosa
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel, AutoFeatureExtractor
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import torch.nn as nn
import math
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model, WavLMModel,WavLMConfig


class Melspec(torch_nn.Module):
    """ Mel-spectrogram
    """
    def __init__(self):
        super(Melspec, self).__init__()

    def forward(self, x):
        melspec = librosa.feature.melspectrogram(y=x[0].numpy(), sr=16000, n_fft=512, hop_length=128)
        return torch.from_numpy(melspec)
class rolloff(torch_nn.Module):
    """ Mel-spectrogram
    """
    def __init__(self):
        super(rolloff, self).__init__()

    def forward(self, x):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x[0].numpy(), n_fft= 512, hop_length=128, sr=16000, roll_percent=0.75)
        return torch.from_numpy(spectral_rolloff)





class XLSR(torch.nn.Module):
    def __init__(self, model_dir, device='cuda', sampling_rate=16000, freeze=True, visual=False):
        super(XLSR, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)
        self.freeze = freeze
        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.visual = visual
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)
        # print(self.model)
        if self.visual:
            outputs = self.model(feat, output_attentions=self.visual)
            last_hidden_state = outputs.last_hidden_state
            attentions = outputs.attentions
            return last_hidden_state, attentions
        if self.freeze:
            with torch.no_grad():
                output = self.model(feat).last_hidden_state
                
        else:
            output = self.model(feat).last_hidden_state

        return output
    
    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output



class WAVLM(torch.nn.Module):
    def __init__(self, model_dir, device='cuda', sampling_rate=16000, freeze=True):
        super(WAVLM, self).__init__()
        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = WavLMConfig.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = WavLMModel.from_pretrained(model_dir).to(self.device)
        self.freeze = freeze
        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        if self.freeze:
            with torch.no_grad():
                output = self.model(feat).last_hidden_state
        else:
            output = self.model(feat).last_hidden_state
        return output
    
    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output



class MERT(torch.nn.Module):
    def __init__(self, model_dir, device='cuda', sampling_rate=16000, freeze=True):
        super(MERT, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir, sampling_rate = 16000,do_normalize = False)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
        self.freeze = freeze
        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        if self.freeze:
            with torch.no_grad():
                output = self.model(feat).last_hidden_state
        else:
            output = self.model(feat).last_hidden_state
        return output
    
    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output


class PT_XLSR(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PT_XLSR, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens  # Assume prompt consists of 10 tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim))  # 24 layers
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.visual = visual
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            feat = feat.transpose(1, 2)
            # Feature projection
            hidden_state, extract_features = self.model.feature_projection(feat)
            if self.visual:
                first_hidden_state = hidden_state
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_state)
            hidden_state = hidden_state + position_embeddings
            hidden_state = self.model.encoder.dropout(hidden_state)
        if self.visual:
            all_self_attentions = []
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                hidden_state = torch.cat((prompt, hidden_state), dim=1)
                if self.visual:
                    print(hidden_state.shape, 'hidden_state')
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  
                hidden_state = torch.cat((prompt, hidden_state[:, self.num_prompt_tokens:, :]), dim=1)
                if self.visual: 
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)  
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]  

        if self.visual:
            print(len(all_self_attentions), "all_self_attentions")
            return first_hidden_state, hidden_state,all_self_attentions
        else:
            return hidden_state
    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output

    
class PT_WAVLM(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PT_WAVLM, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = WavLMConfig.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = WavLMModel.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens  # Assume prompt consists of 10 tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim))  # 24 layers
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.visual = visual
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        with torch.no_grad():
            outputs = self.model(feat)
            hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
        hidden_state = hidden_states[0]
            
          
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                hidden_state = torch.cat((prompt, hidden_state), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  
                hidden_state = torch.cat((prompt, hidden_state[:, self.num_prompt_tokens:, :]), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]

        if self.visual:
            encoder_outputs = self.model.encoder(
            hidden_states=hidden_state,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

            attention_weights = encoder_outputs.attentions
            return hidden_state,attention_weights
        else:
            return hidden_state

    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output


class PT_MERT(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PT_MERT, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir, sampling_rate = 16000,do_normalize = False)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens  # Assume prompt consists of 10 tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim))  # 24 layers
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.visual = visual
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        with torch.no_grad():
            outputs = self.model(feat)
            hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
        hidden_state = hidden_states[0]
          
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                hidden_state = torch.cat((prompt, hidden_state), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  
                hidden_state = torch.cat((prompt, hidden_state[:, self.num_prompt_tokens:, :]), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]

        if self.visual:
            encoder_outputs = self.model.encoder(
            hidden_states=hidden_state,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

            attention_weights = encoder_outputs.attentions
            return hidden_state,attention_weights
        else:
            return hidden_state

    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output





    
if __name__ == "__main__":
    wav = torch.randn(16, 64600)

    model_dir = "/data3/xyk/huggingface/wav2vec2-xls-r-300m/"

    model = XLSRsplittest(prompt_dim=1024, model_dir=model_dir, device='cuda')
    combined_features = model.extract_features(wav)
    print(combined_features.shape,'combined_features')

