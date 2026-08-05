# Based on NVIDIA Isaac Sim's official tools/docker/web-viewer/Dockerfile.
FROM node:22-bookworm-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN echo '@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/' > /root/.npmrc \
    && npx @nvidia/create-ov-web-rtc-app@1.14.2 \
        --name web-viewer --sample local-sample --outputDir /app \
    && rm -f /root/.npmrc

WORKDIR /app/web-viewer
ARG ISAACSIM_HOST=127.0.0.1
ARG ISAACSIM_SIGNAL_PORT=49100
ARG ISAACSIM_STREAM_PORT=47998

RUN sed -i "s/signalingServer: [^,]*/signalingServer: '${ISAACSIM_HOST}'/" src/main.ts \
    && sed -i "/signalingServer:/a\\    signalingPort: ${ISAACSIM_SIGNAL_PORT},\n    mediaServer: '${ISAACSIM_HOST}',\n    mediaPort: ${ISAACSIM_STREAM_PORT},\n    forceWSS: false," src/main.ts \
    && npm install --ignore-scripts \
    && npm run build

FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/web-viewer/package.json ./
RUN echo '@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/' > /root/.npmrc \
    && npm install --ignore-scripts \
    && npm cache clean --force \
    && rm -f /root/.npmrc
COPY --from=builder /app/web-viewer/dist ./dist
USER node
EXPOSE 8210
CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "8210"]
