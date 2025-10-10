#!/bin/bash
# ============================================
# ⚡ Setup para acelerar build no Streamlit Cloud
# ============================================

echo "🚀 Pré-instalando dependências principais..."

# Atualiza o sistema e pip
pip install --upgrade pip setuptools wheel

# Instala pacotes pesados em cache antes do requirements.txt
pip install numpy scipy pandas scikit-learn

# Confirma instalação
python -m pip list

echo "✅ Ambiente pré-configurado com sucesso!"
echo "Agora você pode prosseguir com a instalação das dependências do projeto"
echo "Use: pip install -r requirements.txt"
echo "Boa sorte! 🍀"