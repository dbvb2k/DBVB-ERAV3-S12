from flask import Flask, render_template, request, jsonify
import torch
import tiktoken
from train_get2_8_init import GPT, GPTConfig
import os

app = Flask(__name__)

# Global variables to store model and tokenizer
model = None
tokenizer = None
device = None

def init_model_and_tokenizer():
    """Initialize model and tokenizer if not already initialized"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        try:
            tokenizer = tiktoken.get_encoding('gpt2')
            model, device = load_model()
        except Exception as e:
            print(f"Failed to initialize model or tokenizer: {str(e)}")
            raise

def load_model():
    try:
        # Initialize model with same config as training
        config = GPTConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            vocab_size=50257
        )
        model = GPT(config)
        
        # Check if model file exists
        model_path = 'best_model.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found")
        
        # Load the trained weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load state dict with error handling
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Handle both full checkpoint and state dict only cases
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Attempt to load state dict and catch specific error
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Error loading model weights: {str(e)}")
            print("Available keys in state_dict:", state_dict.keys())
            print("Model's state_dict keys:", model.state_dict().keys())
            
            # Try loading with strict=False as fallback
            print("Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("Loaded with missing/unexpected keys")
            
        model.to(device)
        model.eval()
        return model, device
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

def generate_text(prompt, max_length=50, num_predictions=5, temperature=0.8):
    global model, tokenizer, device
    
    # Ensure model and tokenizer are initialized
    if model is None or tokenizer is None:
        init_model_and_tokenizer()
    
    # Tokenize the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_predictions):
            current_input = input_ids.clone()
            
            for _ in range(max_length):
                # Get model's prediction
                logits, _ = model(current_input)
                logits = logits[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input
                current_input = torch.cat([current_input, next_token], dim=1)
            
            # Decode the generated sequence
            generated_tokens = current_input[0].tolist()[len(input_ids[0]):]
            generated_text = tokenizer.decode(generated_tokens)
            predictions.append(generated_text)
    
    return predictions

@app.route('/')
def home():
    # Initialize model on first request
    if model is None:
        init_model_and_tokenizer()
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        
        # Validate prompt
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
            
        # Validate and convert numeric parameters with bounds
        try:
            num_predictions = int(data.get('num_predictions', 5))
            if not 1 <= num_predictions <= 10:
                return jsonify({'error': 'Number of predictions must be between 1 and 10'}), 400
                
            max_length = int(data.get('max_length', 50))
            if not 1 <= max_length <= 200:
                return jsonify({'error': 'Maximum length must be between 1 and 200'}), 400
                
            temperature = float(data.get('temperature', 0.8))
            if not 0.1 <= temperature <= 2.0:
                return jsonify({'error': 'Temperature must be between 0.1 and 2.0'}), 400
                
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid numeric parameters provided'}), 400
        
        predictions = generate_text(
            prompt=prompt,
            max_length=max_length,
            num_predictions=num_predictions,
            temperature=temperature
        )
        
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 