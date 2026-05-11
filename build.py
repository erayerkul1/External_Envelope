"""
Çalıştır:  python build.py
Çıktı:     dist/ExternalEnvelope.exe   ← buna çift tıkla
"""
import subprocess
import sys

subprocess.run(
    [
        sys.executable, "-m", "PyInstaller",
        "--onefile",       # tek .exe dosyası
        "--windowed",      # konsol penceresi açma
        "--name", "ExternalEnvelope",
        "envelope.py",
    ],
    check=True,
)
print("\nTamamlandı! dist/ExternalEnvelope.exe dosyasına çift tıklayabilirsiniz.")
