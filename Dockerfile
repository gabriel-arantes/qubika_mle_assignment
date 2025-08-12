# Usa uma imagem base oficial do Python
FROM python:3.13-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Define variáveis de ambiente para otimizar a execução do Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copia o arquivo de dependências primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências listadas no requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia todo o restante do código do projeto para o diretório de trabalho no container
COPY . .

# Expõe a porta em que a API será executada
EXPOSE 8000

# Define o comando para iniciar a aplicação quando o container for executado
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]