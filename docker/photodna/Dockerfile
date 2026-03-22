# Force x86_64 so wine can run Windows x86_64 binaries (Wine Python + PhotoDNA DLL).
# On Apple Silicon, Docker Desktop uses Rosetta 2 for x86_64 emulation.
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV WINEDEBUG=-all

RUN apt-get update -qq \
 && apt-get install -y -q --no-install-recommends \
        wine64 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Wine Python 3.9 embeddable (64-bit Windows Python that runs via wine64)
RUN mkdir -p /app/python \
 && curl -fsSL \
    https://github.com/jankais3r/pyPhotoDNA/releases/download/wine_python_39/wine_python_39.tar.gz \
    | tar xz -C /app/python --strip-components=1

WORKDIR /app
COPY PhotoDNAx64.dll /app/PhotoDNAx64.dll
COPY generateHashes.py /app/generateHashes.py

# Persistent stdin/stdout server: reads base64 PNG lines, writes CSV hash lines.
# Started once per process with: docker run --rm -i --platform linux/amd64 evohash-photodna
RUN ln -s /usr/bin/wine64-stable /usr/local/bin/wine64

ENTRYPOINT ["wine64", "/app/python/python.exe", "/app/generateHashes.py"]
