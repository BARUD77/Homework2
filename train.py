import torch, math
from pathlib import Path

def train_model_simple(
    model, gpt2, train_loader, val_loader, optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer,
    # --- new ---
    early_stop=True, patience=5, min_delta=1e-3,
    save_path="best_model.pt",
    use_plateau_lr=True, lr_factor=0.5, lr_patience=2, min_lr=1e-6,
    max_tokens_seen=None  # e.g., 10_000_000 to cap training by token budget
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val = math.inf
    bad_count = 0
    stopped_early = False

    # optional LR scheduler on validation loss
    scheduler = None
    if use_plateau_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor,
            patience=lr_patience, threshold=min_delta, min_lr=min_lr
        )

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # safe default
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # eval
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train {train_loss:.3f} | Val {val_loss:.3f} | tokens {tokens_seen:,}")

                # scheduler step on val loss
                if scheduler is not None:
                    scheduler.step(val_loss)

                # early stopping logic
                if early_stop:
                    if val_loss < (best_val - min_delta):
                        best_val = val_loss
                        bad_count = 0
                        if save_path is not None:
                            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                            torch.save({'model': model.state_dict()}, save_path)
                            print(f"  ↳ new best; saved to {save_path}")
                    else:
                        bad_count += 1
                        print(f"  ↳ no improvement ({bad_count}/{patience})")
                        if bad_count >= patience:
                            print("Early stopping triggered.")
                            stopped_early = True
                            break  # break batch loop

            # optional token budget
            if (max_tokens_seen is not None) and (tokens_seen >= max_tokens_seen):
                print("Reached token budget; stopping.")
                stopped_early = True
                break

        # sample after each epoch
        generate_and_print_sample(model, gpt2, tokenizer, device, start_context)

        if stopped_early:
            break

    # load best model back, if saved
    if early_stop and save_path is not None and Path(save_path).exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded best model from {save_path} (val loss {best_val:.3f}).")

    return train_losses, val_losses, track_tokens_seen



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, gpt2, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(gpt2, start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss    


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# def generate_text_simple(model, idx, max_new_tokens, context_size):
#     # idx is (B, T) array of indices in the current context

#     for _ in range(max_new_tokens):

#         # Crop current context if it exceeds the supported context size
#         # E.g., if LLM supports only 5 tokens, and the context size is 10
#         # then only the last 5 tokens are used as context
#         idx_cond = idx[:, -context_size:]

#         # Get the predictions
#         with torch.no_grad():
#             logits = model(idx_cond)

#         # Focus only on the last time step
#         # (batch, n_token, vocab_size) becomes (batch, vocab_size)
#         logits = logits[:, -1, :]

#         # Get the idx of the vocab entry with the highest logits value
#         idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

#         # Append sampled index to the running sequence
#         idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

#     return idx

def text_to_token_ids(gpt2, text, tokenizer):
  if gpt2:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  else:
    encoded = tokenizer.encode(text)
  encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
  return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
  flat = token_ids.squeeze(0) # remove batch dimension
  return tokenizer.decode(flat.tolist())

import torch
import torch.nn.functional as F

def sample_top_k(logits, k=50, temperature=0.9):
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)
    if k is not None:
        v, _ = torch.topk(logits, k)
        cutoff = v[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0.9, top_k=50):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]   # last step
        idx_next = sample_top_k(logits, k=top_k, temperature=temperature)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
