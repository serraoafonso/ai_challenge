
"""
recebe a imagem e devolve se é maligno ou benigno confrome o modelo ja treinado
"""


from fastapi import FastAPI, Request #para receber imagens
from fastapi.responses import JSONResponse  
import uvicorn      #servidor                          
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image                     
import io                                   


app = FastAPI()  


class SimpleCNN(nn.Module):
    """
    Cérebro que aprendeu a distinguir de maligno de benigno
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: benigno / maligno

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)  # achata a imagem
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# carrega os pesos do modelo que treinamos antes
model.load_state_dict(torch.load("breast_model.pth", map_location=device))
model.eval()  # Diz ao modelo: "não vamos treinar mais, só adivinhar"

#transformar imagem
transform = transforms.Compose([
    transforms.Resize((28, 28)),        
    transforms.ToTensor(),               
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

#rota que recebe as imagens, porta 5000
@app.post("/predict")  # Quando alguém fizer POST /predict
async def predict(request: Request):
    try:
        body = await request.body()
        image = Image.open(io.BytesIO(body)).convert("L")  # grayscale

        img_tensor = transform(image).unsqueeze(0).to(device)

        # Fazer previsão sem alterar o modelo
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            label = "Benigno" if predicted.item() == 0 else "Maligno"

        # Devolver a resposta para o frontend
        return JSONResponse(content={"prediction": label})

    except Exception as e:
        # Se algo der errado, devolve erro
        return JSONResponse(content={"error": str(e)}, status_code=500)

#roda o servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
