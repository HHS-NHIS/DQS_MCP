FROM node:20-alpine
WORKDIR /app
COPY package.json ./
RUN npm install --production
COPY server.js ./server.js
COPY index.html ./index.html
ENV PORT=8099
EXPOSE 8099
CMD ["node","server.js"]
