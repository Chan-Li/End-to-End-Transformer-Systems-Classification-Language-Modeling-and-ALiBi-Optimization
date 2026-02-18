from email import utils
from json import decoder
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
import tokenizer
import argparse
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerClassifier, TransformerDecoderAlibi, TransformerDecoderLM
from utilities import Utilities
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, attn_maps = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
        # Set up argument parser
    parser = argparse.ArgumentParser(description='Run three parts')
    parser.add_argument('--part', type=str, required=True, help='which part to run: CLS, LM, or the exploration')
    args = parser.parse_args()
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    if args.part == "one":
        print("Running part 1: CLS")
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(
        test_CLS_dataset,
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=False)
        torch.manual_seed(seed)
        encoder = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        n_hidden=n_hidden,
        n_output=n_output).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
        print("\n=== Training classifier (end-to-end encoder + classifier) ===")
        # for the classification  task, you will train for a fixed number of epochs like this:
        for epoch in range(epochs_CLS):
            encoder.train()
            running_loss = 0.0

            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.long()
                yb = yb.long()

                logits, _ = encoder(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

            avg_loss = running_loss / len(train_CLS_dataset)

            train_acc = compute_classifier_accuracy(encoder, train_CLS_loader)
            test_acc  = compute_classifier_accuracy(encoder, test_CLS_loader)

            print(
                f"Epoch {epoch+1:02d}/{epochs_CLS} "
                f"| loss={avg_loss:.4f} "
                f"| train_error={train_acc:.2f}%"
                f"| test_error={test_acc:.2f}%"
            )

        print("\n=== Running sanity check on a training example ===")
        sample_ids, _ = test_CLS_dataset[0]
        sentence = tokenizer.decode(sample_ids.tolist())
        print("Sentence used for sanity check:")
        print(sentence)
        encoder.to("cpu")
        utils = Utilities(tokenizer, encoder)
        utils.sanity_check(sentence, block_size)
        total_params = sum(p.numel() for p in encoder.parameters())
        print("Total parameters:", total_params)
    # CLS training code here
    if args.part == "two":
        print("Running part 2: LM")

        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        inputfile2 = "speechesdataset/test_LM_obama.txt"
        with open(inputfile2, 'r', encoding='utf-8') as f:
            lmtestText_obama = f.read()
        inputfile3 = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile3, 'r', encoding='utf-8') as f:
            lmtestText_wbush = f.read()
        inputfile4 = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile4, 'r', encoding='utf-8') as f:
            lmtestText_ghbush = f.read()

        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        # --- Build test datasets ---
        test_obama_dataset = LanguageModelingDataset(tokenizer, lmtestText_obama, block_size)
        test_wbush_dataset = LanguageModelingDataset(tokenizer, lmtestText_wbush, block_size)
        test_ghbush_dataset = LanguageModelingDataset(tokenizer, lmtestText_ghbush, block_size)

        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=False)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=False)
        test_ghbush_loader = DataLoader(test_ghbush_dataset, batch_size=batch_size, shuffle=False)

 

        decoder = TransformerDecoderLM(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_hidden=100,
            return_attn_maps=True
        ).to(device)
        optimizer_lm = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

        print("\n=== Training decoder (language modeling) ===")
        decoder.train()
        running = 0.0
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.long()
            yb = yb.long()

            loss, attn_maps = decoder(xb, yb)

            optimizer_lm.zero_grad()
            loss.backward()
            optimizer_lm.step()

            running += loss.item()

            if (i + 1) % eval_interval == 0:
                avg_loss = running / eval_interval

                obama_ppl = compute_perplexity(decoder, test_obama_loader, eval_iters=eval_iters)
                wbush_ppl = compute_perplexity(decoder, test_wbush_loader, eval_iters=eval_iters)
                ghbush_ppl = compute_perplexity(decoder, test_ghbush_loader, eval_iters=eval_iters)

                print( f"Iter {i+1:04d}/{max_iters} "
                # f"| train_ppl={train_ppl:.2f} "
                f"| obama_ppl={obama_ppl:.2f} "
                f"| wbush_ppl={wbush_ppl:.2f} "
                f"| ghbush_ppl={ghbush_ppl:.2f}")
            
        print("\n=== Final Perplexity Evaluation ===")
        train_ppl_final = compute_perplexity(
                decoder,
                train_LM_loader,
                eval_iters=eval_iters
            )
        obama_ppl = compute_perplexity(decoder, test_obama_loader, eval_iters=eval_iters)
        wbush_ppl = compute_perplexity(decoder, test_wbush_loader, eval_iters=eval_iters)
        ghbush_ppl = compute_perplexity(decoder, test_ghbush_loader, eval_iters=eval_iters)

        print(f"Final Train Perplexity : {train_ppl_final:.2f}")
        print(f"Obama  Perplexity      : {obama_ppl:.2f}")
        print(f"W Bush Perplexity      : {wbush_ppl:.2f}")
        print(f"GH Bush Perplexity     : {ghbush_ppl:.2f}")
        total_params = sum(p.numel() for p in decoder.parameters())
        print("Total parameters:", total_params)
        
        print("\n=== Running sanity check on GH Bush test sentence ===")
        first_sentence = lmtestText_ghbush.split(".")[0] + "."
        print("Sentence used for sanity check:")
        print(first_sentence)

        decoder_cpu = decoder.to("cpu")
        utils = Utilities(tokenizer, decoder_cpu)
        utils.sanity_check(first_sentence, block_size)

        decoder = decoder.to(device)
            # LM training code here
    if args.part == "three":
        print("Running part 3: ALiBi positional encoding")
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        inputfile2 = "speechesdataset/test_LM_obama.txt"
        with open(inputfile2, 'r', encoding='utf-8') as f:
            lmtestText_obama = f.read()
        inputfile3 = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile3, 'r', encoding='utf-8') as f:
            lmtestText_wbush = f.read()
        inputfile4 = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile4, 'r', encoding='utf-8') as f:
            lmtestText_ghbush = f.read()

        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        # --- Build test datasets ---
        test_obama_dataset = LanguageModelingDataset(tokenizer, lmtestText_obama, block_size)
        test_wbush_dataset = LanguageModelingDataset(tokenizer, lmtestText_wbush, block_size)
        test_ghbush_dataset = LanguageModelingDataset(tokenizer, lmtestText_ghbush, block_size)

        test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=False)
        test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=False)
        test_ghbush_loader = DataLoader(test_ghbush_dataset, batch_size=batch_size, shuffle=False)

 

        decoder = TransformerDecoderAlibi(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_hidden=100,
            return_attn_maps=True
        ).to(device)
        optimizer_lm = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

        print("\n=== Training decoder (language modeling) ===")
        decoder.train()
        running = 0.0
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.long()
            yb = yb.long()

            loss, attn_maps = decoder(xb, yb)

            optimizer_lm.zero_grad()
            loss.backward()
            optimizer_lm.step()

            running += loss.item()

            if (i + 1) % eval_interval == 0:
                avg_loss = running / eval_interval

                obama_ppl = compute_perplexity(decoder, test_obama_loader, eval_iters=eval_iters)
                wbush_ppl = compute_perplexity(decoder, test_wbush_loader, eval_iters=eval_iters)
                ghbush_ppl = compute_perplexity(decoder, test_ghbush_loader, eval_iters=eval_iters)

                print( f"Iter {i+1:04d}/{max_iters} "
                # f"| train_ppl={train_ppl:.2f} "
                f"| obama_ppl={obama_ppl:.2f} "
                f"| wbush_ppl={wbush_ppl:.2f} "
                f"| ghbush_ppl={ghbush_ppl:.2f}")
            
        print("\n=== Final Perplexity Evaluation ===")
        train_ppl_final = compute_perplexity(
                decoder,
                train_LM_loader,
                eval_iters=eval_iters
            )
        obama_ppl = compute_perplexity(decoder, test_obama_loader, eval_iters=eval_iters)
        wbush_ppl = compute_perplexity(decoder, test_wbush_loader, eval_iters=eval_iters)
        ghbush_ppl = compute_perplexity(decoder, test_ghbush_loader, eval_iters=eval_iters)

        print(f"Final Train Perplexity : {train_ppl_final:.2f}")
        print(f"Obama  Perplexity      : {obama_ppl:.2f}")
        print(f"W Bush Perplexity      : {wbush_ppl:.2f}")
        print(f"GH Bush Perplexity     : {ghbush_ppl:.2f}")
            # You can put any code you want here for exploration. You can train a different model, or do some analysis on the models you trained in part 1 and part 2, or something else. This is your chance to be creative and explore something interesting related to transformers. You can also use this space to do some hyperparameter tuning if you want. Just make sure to include a description of what you are doing and why in the comments.
    

        



if __name__ == "__main__":
    main()
