#!/usr/bin/env bash

# Set the name of your virtual machine
NAME="rag8s"

# Define the path to your SSH configuration directory and file
SSH_DIR="$HOME/.ssh"
SSH_CONFIG="$SSH_DIR/config"

# Ensure the SSH directory exists and has the correct permissions
echo "Ensuring $SSH_DIR exists..."
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Start the virtual machine
echo "Starting VM: $NAME..."
vagrant up

# Reload the VM to apply any changes
echo "Reloading VM to apply Docker group changes..."
vagrant reload --provision

# Clean up any existing SSH configuration for the VM
echo "Cleaning old SSH block for $NAME..."
if [ -f "$SSH_CONFIG" ]; then
  sed -i.bak "/^Host $NAME\$/,/^Host /{/^Host $NAME\$/!{/^Host /!d}}" "$SSH_CONFIG"
fi

# Add the new SSH configuration for the VM
echo "Adding fresh SSH config for $NAME..."
vagrant ssh-config | sed "s/^Host default/Host $NAME/" >> "$SSH_CONFIG"

# Set the correct permissions for the SSH configuration file
chmod 600 "$SSH_CONFIG"

# Extract the path to the private key from the SSH configuration
KEY_PATH=$(awk "/^Host $NAME\$/,/^Host /{ if (/IdentityFile/) print \$2 }" "$SSH_CONFIG" | head -n1)
if [[ -n "$KEY_PATH" && -f "$KEY_PATH" ]]; then
  chmod 600 "$KEY_PATH"
  echo "✔️ Secured $KEY_PATH"
else
  echo "⚠️ Warning: No private key found for $NAME"
fi

# Install the VS Code Remote SSH extension
echo "Installing VS Code Remote SSH extension..."
code --install-extension ms-vscode-remote.remote-ssh

# Apply custom VS Code settings
echo "Applying custom VS Code settings..."
USER_SETTINGS="$HOME/.config/Code/User/settings.json"
CUSTOM_SETTINGS="$HOME/.vscode-setup/settings.json"
if [ -f "$CUSTOM_SETTINGS" ]; then
  cp "$CUSTOM_SETTINGS" "$USER_SETTINGS"
  echo "✔️ Applied custom VS Code settings."
else
  echo "⚠️ Custom settings file not found: $CUSTOM_SETTINGS"
fi

# Verify the SSH connection
echo "Verifying SSH..."
ssh -q "$NAME" exit && echo "✔️ SSH works!" || { echo "❌ SSH failed."; exit 1; }

# Open VS Code with the Remote SSH connection
echo "Opening VS Code with Remote SSH: $NAME"
code --folder-uri "vscode-remote://ssh-remote+$NAME/vagrant"

