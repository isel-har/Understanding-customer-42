

def train(model, criterion, optimizer, dataloader):
    train_loss = 0
    train_correct = 0

    for X_batch, y_batch in dataloader:

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(dim=1) == y_batch).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc      = train_correct / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/20] Loss: {train_loss/len(train_loader):.4f}")

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.3f}"
    )
