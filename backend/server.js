// server.js
import express from "express";
import multer from "multer";
import cors from "cors";
import fetch from "node-fetch"; 

const app = express();
const PORT = 8000;

app.use(cors());

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.post("/predict", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Nenhum ficheiro enviado" });
    }

    // envia para o serviço Python (FastAPI)
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: req.file.buffer,//representacao do arquivo em memoria
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Filename": req.file.originalname,
      },
    });

    const data = await response.json();
    res.json(data); // devolve ao frontend
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Erro ao comunicar com o serviço Python" });
  }
});

app.listen(PORT, () => {
  console.log(`Servidor Node a correr em http://localhost:${PORT}`);
});
