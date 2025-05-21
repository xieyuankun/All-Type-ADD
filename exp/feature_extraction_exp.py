import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from pytorch_wavelets import DWTForward,DWT1DForward
import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel, AutoFeatureExtractor, WavLMModel,WavLMConfig

class WaveletBlock(nn.Module):
    def __init__(self, wave='haar', J=2, input_dim=1024, output_dim=1024):
        super(WaveletBlock, self).__init__()
        self.dwt = DWTForward(J=J, wave=wave)  
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        """
        input: (batch, token, dim)
        output: (batch, token, dim)
        """
        B, T, D = x.shape
        assert D == self.input_dim, f"Input dimension (dim={D}) must match WaveletBlock's input_dim ({self.input_dim})"

        x = x.unsqueeze(dim=1) # channel 1
        # wavelet transform
        LL, band = self.dwt(x)  #  LL([b, 1, 3, 512]) band （LH/HL/HH）([b, 1, 3, 3, 512])
        bands= band[0]  
        LL = LL.unsqueeze(dim=2) #  LL([b, 1, 3, 1, 512]) 
        # print(bands.shape, 'bands')
        # print(LL.shape,'LL')
        features = torch.cat((LL, bands), dim=2).view(B, -1, D)# (batch, token, output_dim)
        return features


class WPT_XLSR(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=5, num_wavelet_tokens = 6
                 , dropout=0.1,visual=False):
        super(WPT_XLSR, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.visual = visual    
        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens 
        self.num_wavelet_tokens = num_wavelet_tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim)) 
        self.fprompt_embedding = nn.Parameter(torch.zeros(24, self.num_wavelet_tokens, prompt_dim)) 
        self.wavelet_block = WaveletBlock(wave='haar', J=1, input_dim=1024, output_dim=1024)
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        nn.init.uniform_(self.fprompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        
        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            feat = feat.transpose(1, 2)
            # Feature projection
            hidden_state, extract_features = self.model.feature_projection(feat)
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_state)
            hidden_state = hidden_state + position_embeddings
            hidden_state = self.model.encoder.dropout(hidden_state) # equal to hidden_state = hidden_states[0]

        B = feat.size(0)  
        if self.visual:
            all_self_attentions = []
            
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt,prompt, hidden_state), dim=1)
                if self.visual:
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  # Apply dropout to prompt
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt, prompt, hidden_state[:, self.num_prompt_tokens + fprompt.shape[1]:, :]), dim=1)
                if self.visual: 
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)  
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]    
                    
        if self.visual:
            print(len(all_self_attentions), "all_self_attentions")
            return hidden_state,all_self_attentions
        else:
            print(hidden_state.shape,'hidden_state')
            return hidden_state


    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output











class WPT_WAVLM(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=5, num_wavelet_tokens = 6
                 , dropout=0.1,visual=False):
        super(WPT_WAVLM, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = WavLMConfig.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize = False)
        self.model = WavLMModel.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.visual = visual    
        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens 
        self.num_wavelet_tokens = num_wavelet_tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim)) 
        self.fprompt_embedding = nn.Parameter(torch.zeros(24, self.num_wavelet_tokens, prompt_dim)) 
        self.wavelet_block = WaveletBlock(wave='haar', J=1, input_dim=1024, output_dim=1024)
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        nn.init.uniform_(self.fprompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        
        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            feat = feat.transpose(1, 2)
            # Feature projection
            hidden_state, extract_features = self.model.feature_projection(feat)
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_state)
            hidden_state = hidden_state + position_embeddings
            hidden_state = self.model.encoder.dropout(hidden_state)
        
          # This is the input of the transformer layer 
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt,prompt, hidden_state), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  # Apply dropout to prompt
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt, prompt, hidden_state[:, self.num_prompt_tokens + self.num_wavelet_tokens:, :]), dim=1)
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
    
class WPT_MERT(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=5, num_wavelet_tokens = 6
                 , dropout=0.1,visual=False):
        super(WPT_MERT, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir, sampling_rate = 16000,do_normalize = False)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.visual = visual    
        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens 
        self.num_wavelet_tokens = num_wavelet_tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim)) 
        self.fprompt_embedding = nn.Parameter(torch.zeros(24, self.num_wavelet_tokens, prompt_dim)) 
        self.wavelet_block = WaveletBlock(wave='haar', J=1, input_dim=1024, output_dim=1024)
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        nn.init.uniform_(self.fprompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        
        with torch.no_grad():
            outputs = self.model(feat)
            hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
        hidden_state = hidden_states[0]
        
          # This is the input of the transformer layer 
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt,prompt, hidden_state), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  # Apply dropout to prompt
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt, prompt, hidden_state[:, self.num_prompt_tokens + self.num_wavelet_tokens:, :]), dim=1)
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
    
    
        
class PT_XLSR_shallow(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1):
        super(PT_XLSR_shallow, self).__init__()

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
        self.prompt_embedding = nn.Parameter(torch.zeros(1, self.num_prompt_tokens, prompt_dim))  # 24 layers
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            feat = feat.transpose(1, 2)
            # Feature projection
            hidden_state, extract_features = self.model.feature_projection(feat)
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_state)
            hidden_state = hidden_state + position_embeddings
            hidden_state = self.model.encoder.dropout(hidden_state)
          
        B = feat.size(0)  
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                hidden_state = torch.cat((prompt, hidden_state), dim=1)
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
        return hidden_state

    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output    
    
    
if __name__ == "__main__":

    # wavlet = WaveletBlock().cuda()
    # prompt = torch.randn(16, 5, 1024).cuda()
    # features = wavlet(prompt)
    # print(features.shape)
    wav = torch.ones(2, 64600).cuda()
    model = WPT_XLSR(model_dir='/data3/xyk/huggingface/wav2vec2-xls-r-300m/', prompt_dim=1024, num_prompt_tokens=5, num_wavelet_tokens=4).cuda()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_in_million = trainable_params / 1e6

    print(f"Trainable parameters (in million): {trainable_params_in_million:.2f}M")

    features = model(wav)