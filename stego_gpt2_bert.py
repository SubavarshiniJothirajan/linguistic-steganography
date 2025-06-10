# Install required libraries (uncomment if needed)
# !pip install torch transformers numpy

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertForMaskedLM
import numpy as np
import heapq
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

def text_to_binary(secret_text):
    binary = ''.join(format(ord(char), '08b') for char in secret_text)
    return binary

def binary_to_text(binary):
    if len(binary) % 8 != 0:
        print(f"Warning: Binary length {len(binary)} is not a multiple of 8")
    binary = binary[:len(binary) - (len(binary) % 8)]  # Truncate to multiple of 8
    chars = [chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8)]
    return ''.join(chars)

def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    huffman_tree = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    return huffman_tree

def generation_unit(prefix, secret_bits, max_length=50, T=5.0, BN=3):
    input_ids = gpt2_tokenizer.encode(prefix, return_tensors='pt').to(device)
    generated_text = prefix.split()
    embedding_record = []
    secret_index = 0
    original_probs = []
    stego_probs = []

    with torch.no_grad():
        for step in range(max_length - len(prefix.split())):
            outputs = gpt2_model(input_ids)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            original_probs.append(probabilities.copy())

            sorted_probs = np.sort(probabilities)[::-1]
            max_prob = sorted_probs[0]
            second_max_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
            D = max_prob / second_max_prob if second_max_prob > 0 else float('inf')
            print(f"Step {step} - D: {D:.4f}, Max Prob: {max_prob:.4f}, Second Max Prob: {second_max_prob:.4f}")

            if secret_index < len(secret_bits) and D <= T:
                candidate_indices = np.argsort(probabilities)[-2**BN:]
                candidate_probs = probabilities[candidate_indices]
                candidate_tokens = gpt2_tokenizer.convert_ids_to_tokens(candidate_indices)
                frequencies = {token: prob for token, prob in zip(candidate_tokens, candidate_probs)}
                huffman_tree = build_huffman_tree(frequencies)
                huffman_mapping = {symbol: code for symbol, code in huffman_tree}
                print(f"  Huffman Mapping: {huffman_mapping}")

                embedded = False
                # Try to embed up to 4 bits (max Huffman code length)
                for code_len in range(4, 0, -1):
                    if secret_index + code_len <= len(secret_bits):
                        target_bits = secret_bits[secret_index:secret_index + code_len]
                        for token, code in huffman_mapping.items():
                            if code == target_bits:
                                next_token_id = gpt2_tokenizer.convert_tokens_to_ids(token)
                                next_token = gpt2_tokenizer.decode([next_token_id], skip_special_tokens=True)
                                embedding_record.append(f"embedded ({code})")
                                secret_index += code_len
                                embedded = True
                                print(f"  Embedding {code} with token '{next_token}' at secret_index {secret_index - code_len}")
                                stego_prob = np.zeros_like(probabilities)
                                stego_prob[next_token_id] = 1.0
                                stego_probs.append(stego_prob)
                                break
                    if embedded:
                        break
                if not embedded:
                    next_token_id = candidate_indices[-1]
                    next_token = gpt2_tokenizer.decode([next_token_id], skip_special_tokens=True)
                    embedding_record.append("not embedded")
                    print(f"  No embedding, using token '{next_token}'")
                    stego_prob = np.zeros_like(probabilities)
                    stego_prob[next_token_id] = 1.0
                    stego_probs.append(stego_prob)
            else:
                next_token_id = np.argmax(probabilities)
                next_token = gpt2_tokenizer.decode([next_token_id], skip_special_tokens=True)
                embedding_record.append("not embedded")
                print(f"  No embedding (D > T or no bits left), using token '{next_token}'")
                stego_prob = np.zeros_like(probabilities)
                stego_prob[next_token_id] = 1.0
                stego_probs.append(stego_prob)

            generated_text.append(next_token)
            new_token_ids = gpt2_tokenizer.encode(next_token, return_tensors='pt').to(device)
            new_token_id = new_token_ids[:, -1:]
            input_ids = torch.cat((input_ids, new_token_id), dim=1)

    intermediate_text = " ".join(generated_text)
    return intermediate_text, embedding_record, original_probs, stego_probs, secret_bits

def substitution_unit(intermediate_text, embedding_record, secret_bits, secret_index, BN=3):
    """Substitute 'not embedded' positions using BERT."""
    tokens = intermediate_text.split()
    final_text_tokens = tokens.copy()
    final_embedding_record = embedding_record.copy()
    current_secret_index = secret_index

    if current_secret_index >= len(secret_bits):
        print("All secret bits already embedded in generation phase. Substitution skipped.")
        return intermediate_text, embedding_record, current_secret_index

    print(f"Starting substitution with {len(secret_bits) - current_secret_index} bits remaining.")
    for i, record in enumerate(final_embedding_record):
        if record == "not embedded" and current_secret_index < len(secret_bits):
            masked_tokens = final_text_tokens.copy()
            masked_tokens[i] = "[MASK]"
            masked_text = " ".join(masked_tokens)
            print(f"Position {i} - Masked Text: {masked_text}")

            inputs = bert_tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True).to(device)
            mask_token_index = torch.where(inputs["input_ids"][0] == bert_tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = bert_model(**inputs)
                logits = outputs.logits[0, mask_token_index, :]
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            candidate_indices = np.argsort(probabilities)[-2**BN:]
            candidate_probs = probabilities[candidate_indices]
            candidate_tokens = [bert_tokenizer.decode([idx]).strip() for idx in candidate_indices]

            frequencies = {token: prob for token, prob in zip(candidate_tokens, candidate_probs)}
            huffman_tree = build_huffman_tree(frequencies)
            huffman_mapping = {symbol: code for symbol, code in huffman_tree}
            print(f"Position {i} - Huffman Mapping: {huffman_mapping}")

            embedded = False
            for code_len in range(4, 0, -1):
                if current_secret_index + code_len <= len(secret_bits):
                    target_bits = secret_bits[current_secret_index:current_secret_index + code_len]
                    for token, code in huffman_mapping.items():
                        if code == target_bits:
                            final_text_tokens[i] = token
                            final_embedding_record[i] = f"embedded ({code}) via substitution"
                            current_secret_index += code_len
                            embedded = True
                            print(f"Position {i} - Embedded {code_len} bits: {code} -> Token: {token}")
                            break
                if embedded:
                    break

            if not embedded:
                final_text_tokens[i] = tokens[i]
                final_embedding_record[i] = "not embedded"
                print(f"Position {i} - No matching Huffman code, retaining: {tokens[i]}")

    final_text = " ".join(final_text_tokens)
    return final_text, final_embedding_record, current_secret_index

def extraction_unit(stego_text, prefix, embedding_record, secret_bits_length, T=5.0, BN=3):
    """Extract secret bits from steganographic text."""
    extracted_bits = []
    current_secret_index = 0
    tokens = stego_text.split()

    for i, record in enumerate(embedding_record):
        if current_secret_index >= secret_bits_length:
            break
        if record.startswith("embedded") and not record.endswith("via substitution"):
            embedded_bits = record.split("(")[1].split(")")[0]
            extracted_bits.append(embedded_bits)
            current_secret_index += len(embedded_bits)
            print(f"Extracted from generation at position {i}: {embedded_bits}")

    for i, (token, record) in enumerate(zip(tokens, embedding_record)):
        if current_secret_index >= secret_bits_length:
            break
        if record.startswith("embedded") and record.endswith("via substitution"):
            masked_tokens = tokens.copy()
            masked_tokens[i] = "[MASK]"
            masked_text = " ".join(masked_tokens)
            inputs = bert_tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True).to(device)
            mask_token_index = torch.where(inputs["input_ids"][0] == bert_tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = bert_model(**inputs)
                logits = outputs.logits[0, mask_token_index, :]
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            candidate_indices = np.argsort(probabilities)[-2**BN:]
            candidate_probs = probabilities[candidate_indices]
            candidate_tokens = [bert_tokenizer.decode([idx]).strip() for idx in candidate_indices]

            frequencies = {token: prob for token, prob in zip(candidate_tokens, candidate_probs)}
            huffman_tree = build_huffman_tree(frequencies)
            huffman_mapping = {symbol: code for symbol, code in huffman_tree}

            if token in huffman_mapping:
                embedded_bits = huffman_mapping[token]
                extracted_bits.append(embedded_bits)
                current_secret_index += len(embedded_bits)
                print(f"Extracted from substitution at position {i}: {embedded_bits}")

    return ''.join(extracted_bits)

def calculate_measures(text, embedding_record, secret_bits, original_probs=None, stego_probs=None):
    """Calculate embedding rate, perplexity, and KL divergence."""
    embedded_bits = 0
    for record in embedding_record:
        if record.startswith("embedded"):
            try:
                bits = record.split("(")[1].split(")")[0]
                embedded_bits += len(bits)
            except IndexError:
                print(f"Warning: Malformed embedding record entry: {record}")
                continue

    sentence_length = len(text.split())
    er = embedded_bits / sentence_length if sentence_length > 0 else 0
    print(f"Embedding Rate (ER): {er:.4f} bits/token")

    input_ids = gpt2_tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = gpt2_model(input_ids)
        logits = outputs.logits[:, :-1, :]
        probs = torch.softmax(logits, dim=-1)
        target_ids = input_ids[:, 1:]
        log_probs = torch.log(probs.gather(2, target_ids.unsqueeze(-1))).squeeze(-1)
        avg_log_prob = log_probs.mean().item()
        ppl = 2 ** (-avg_log_prob / np.log(2))
    print(f"Perplexity (PPL): {ppl:.2f}")

    if original_probs and stego_probs:
        kl_div = 0
        num_steps = len(original_probs)
        for p, q in zip(original_probs, stego_probs):
            kl_div += np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))
        kl_div /= num_steps if num_steps > 0 else 1
        print(f"Average KL Divergence (KL): {kl_div:.4f}")
    else:
        print("KL Divergence: Not calculated")

    print(f"Bits Embedded: {embedded_bits}/{len(secret_bits)}")

def main():
    
    prefix = "The weather today is"
    secret_sentence ="Attack them at night"  # 16 bits
    T = 5.0
    BN = 3
    max_length = 260

    secret_bits = text_to_binary(secret_sentence)
    print(f"\nSecret Sentence: '{secret_sentence}' -> Binary: {secret_bits} ({len(secret_bits)} bits)")

    print("\n=== Generation Unit ===")
    intermediate_text, embedding_record, original_probs, stego_probs, secret_bits = generation_unit(
        prefix, secret_bits, max_length=max_length, T=T, BN=BN
    )
    print("\nGeneration Results:")
    print("Intermediate Text:", intermediate_text)
    print("Embedding Record:", embedding_record)
    calculate_measures(intermediate_text, embedding_record, secret_bits, original_probs, stego_probs)

    secret_index = sum(len(record.split("(")[1].split(")")[0]) for record in embedding_record
                       if record.startswith("embedded") and not record.endswith("via substitution"))
    print(f"\n=== Substitution Unit ===")
    final_text, final_embedding_record, final_secret_index = substitution_unit(
        intermediate_text, embedding_record, secret_bits, secret_index, BN=BN
    )
    print("\nFinal Results:")
    print("Final Steganographic Text:", final_text)
    print("Final Embedding Record:", final_embedding_record)
    calculate_measures(final_text, final_embedding_record, secret_bits)

    print("\n=== Extraction Unit ===")
    extracted_bits = extraction_unit(final_text, prefix, final_embedding_record, len(secret_bits), T=T, BN=BN)
    print(f"Extracted Bits: {extracted_bits}")
    extracted_text = binary_to_text(extracted_bits)
    print(f"Extracted Secret Sentence: '{extracted_text}'")

if __name__ == "__main__":
    main()