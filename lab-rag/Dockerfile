# LabRAG on a NAS or Linux box. Build:  docker build -t labrag ./lab-rag
# Run:   docker run -d --name labrag -p 8008:8008 \
#          -v /mnt/nas/Papers:/papers:ro -v labrag-data:/data -v labrag-models:/root/.labrag/models \
#          -e ANTHROPIC_API_KEY=... labrag
FROM python:3.12-slim
COPY . /src
RUN pip install --no-cache-dir /src && rm -rf /src
ENV LABRAG_FOLDERS=/papers LABRAG_DATA=/data
EXPOSE 8008
CMD ["sh", "-c", "labrag index --every 30 & exec labrag serve"]
