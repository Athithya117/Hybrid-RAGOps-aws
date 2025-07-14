#!/usr/bin/env bash

NAME="RAG8s"
SSH_DIR="$HOME/.ssh"
SSH_CONFIG="$SSH_DIR/config"

echo "Ensuring $SSH_DIR exists..."
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

echo "Starting VM: $NAME..."
vagrant up

echo "Reloading VM to apply Docker group changes..."
vagrant reload --provision

echo "Cleaning old SSH block for $NAME..."
if [ -f "$SSH_CONFIG" ]; then
  sed -i.bak "/^Host $NAME\$/,/^Host /{/^Host $NAME\$/!{/^Host /!d}}" "$SSH_CONFIG"
fi

echo "Adding fresh SSH config for $NAME..."
vagrant ssh-config | sed "s/^Host default/Host $NAME/" >> "$SSH_CONFIG"

chmod 600 "$SSH_CONFIG"

KEY_PATH=$(awk "/^Host $NAME\$/,/^Host /{ if (/IdentityFile/) print \$2 }" "$SSH_CONFIG" | head -n1)
if [[ -n "$KEY_PATH" && -f "$KEY_PATH" ]]; then
  chmod 600 "$KEY_PATH"
  echo "✔️ Secured $KEY_PATH"
else
  echo "⚠️ Warning: No private key found for $NAME"
fi

echo "Installing VS Code Remote SSH extension..."
code --install-extension ms-vscode-remote.remote-ssh

echo "Verifying SSH..."
ssh -q "$NAME" exit && echo "✔️ SSH works!" || { echo "❌ SSH failed."; exit 1; }

echo "Opening VS Code with Remote SSH: $NAME"
code --folder-uri "vscode-remote://ssh-remote+$NAME/vagrant"
