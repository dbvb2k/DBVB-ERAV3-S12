from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import tiktoken
import os
from train_get2_8_init import GPT, GPTConfig

app = Flask(__name__)

# Initialize global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
tokenizer = None

def load_model_and_tokenizer():
    """Load the best model and initialize tokenizer"""
    global model, tokenizer
    
    try:
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding('gpt2')
        
        # Find and load the best model checkpoint
        best_dir = os.path.join('checkpoints', 'best')
        if not os.path.exists(best_dir):
            raise FileNotFoundError("No best model checkpoint found!")
        
        checkpoints = [f for f in os.listdir(best_dir) if f.endswith('.pt')]
        if not checkpoints:
            raise FileNotFoundError("No checkpoint files found!")
        
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(best_dir, latest_checkpoint)
        
        # Load checkpoint and create model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = GPTConfig(**checkpoint['config'])
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Prepare model for inference
        model.to(device)
        model.eval()
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def generate_text(prompt, num_predictions=5, max_length=20, temperature=0.8):
    """Generate text based on prompt"""
    try:
        # Encode the prompt
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        
        # Generate sequences
        generated_sequences = []
        with torch.no_grad():
            for _ in range(num_predictions):
                curr_input_ids = input_ids.clone()
                
                for _ in range(max_length):
                    # Get predictions
                    logits = model(curr_input_ids)[0]
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to sequence
                    curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                
                # Decode and append result
                generated_text = tokenizer.decode(curr_input_ids[0].tolist())
                generated_sequences.append(generated_text)
        
        return generated_sequences
        
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def home():
    global model, tokenizer
    
    # Load model if not loaded
    if model is None or tokenizer is None:
        if not load_model_and_tokenizer():
            return render_template('index.html', 
                                errors=["Failed to load model"])
    
    if request.method == 'POST':
        try:
            # Get and validate inputs
            prompt = request.form.get('prompt', '').strip()
            num_predictions = int(request.form.get('num_predictions', 5))
            max_length = int(request.form.get('max_length', 20))
            temperature = float(request.form.get('temperature', 0.8))
            
            errors = []
            if not prompt:
                errors.append("Please enter some text prompt")
            if num_predictions < 1 or num_predictions > 10:
                errors.append("Number of predictions should be between 1 and 10")
            if max_length < 1 or max_length > 100:
                errors.append("Length should be between 1 and 100")
            if temperature <= 0 or temperature > 2:
                errors.append("Temperature should be between 0 and 2")
                
            if errors:
                return render_template('index.html', errors=errors)
                
            # Generate text
            generated_texts = generate_text(
                prompt=prompt,
                num_predictions=num_predictions,
                max_length=max_length,
                temperature=temperature
            )
            
            return render_template(
                'index.html',
                prompt=prompt,
                generated_texts=generated_texts,
                num_predictions=num_predictions,
                max_length=max_length,
                temperature=temperature
            )
            
        except Exception as e:
            return render_template('index.html', 
                                errors=[f"Error: {str(e)}"])
    
    return render_template('index.html')

if __name__ == '__main__':
    print(f"Using device: {device}")
    if not load_model_and_tokenizer():
        print("Failed to load model! Please check the errors above.")
    app.run(debug=True) 