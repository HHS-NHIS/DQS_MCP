FROM node:20-alpine
WORKDIR /app
COPY package.json ./
RUN npm install --production || true && npm i --omit=dev axios
COPY . .
EXPOSE 8081
CMD ["node","server-http.js"]
