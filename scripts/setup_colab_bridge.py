import os
import subprocess

def setup_tmate():
    print("--- 🛠️ CONFIGURANDO PUENTE TERMINAL (Colab -> Antigravity) ---")
    
    # 1. Install tmate
    subprocess.run(["apt-get", "update", "-y"], capture_output=True)
    subprocess.run(["apt-get", "install", "tmate", "-y"], capture_output=True)
    
    # 2. Launch tmate in background
    # We use a custom socket to avoid conflicts
    print("🚀 Iniciando servidor de túnel...")
    subprocess.Popen(["tmate", "-S", "/tmp/tmate.sock", "new-session", "-d"])
    subprocess.run(["tmate", "-S", "/tmp/tmate.sock", "wait-for", "session-ready"])
    
    # 3. Get SSH connection string
    result = subprocess.run(["tmate", "-S", "/tmp/tmate.sock", "display", "-p", "#{tmate_ssh}"], capture_output=True, text=True)
    ssh_cmd = result.stdout.strip()
    
    print("\n" + "="*50)
    print("🔗 CONEXIÓN LISTA")
    print(f"Copia y pega este comando en tu terminal local (Antigravity):")
    print(f"\033[1;32m{ssh_cmd}\033[0m")
    print("="*50)
    print("\nUna vez conectado, Antigravity podrá tomar el control total.")

if __name__ == "__main__":
    setup_tmate()
