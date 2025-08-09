import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from CatsDogs import CatsDogsDataset #Nuestra clase para cargar el dataset customizado


batch_size = 32
best_val_loss = float('inf') 
epochs_no_improve = 0
patience = 8

#Cargar nuestro corpus (DataSet) y dividirlo en Training , Test y Validación

dataset = CatsDogsDataset(csv_file = "cats_dogs.csv",root_dir = "CatsDogs") #Perro=0 , Gatos=1

print(len(dataset))

train_set,test_set,val_set = torch.utils.data.random_split(dataset,[20000,12461,5000])

#Crear los data loaders
train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)
val_loader = DataLoader(dataset=val_set,batch_size=batch_size,shuffle=False)

# Nos aseguramos que usaremos la GPU, caso por defecto es la CPU
 
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Creación del modelo con nn.Module 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # mantiene tamaño 128x128
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduce a 64x64
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduce a 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduce a 16x16
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512,512), #Capa oculta
            nn.ReLU(), #Activación de la capa oculta
            nn.Linear(512, 2) #Capa final donde clasifica en perro o en gato
        )
    def forward(self,x):
        x = self.conv_layers(x)
        x = self.flatten(x) #Normalización de los datos a un vector
        logits = self.linear_layers(x) #Ejecutamos la red un paso forward con la entrada x
        return logits

model = NeuralNetwork().to(device) #Pasamos a que use la VRAM de la GPU

print(f"El Modelo que hemos creado es {model}")

# Función de perdida que utilizaremos https://pytorch.org/docs/stable/nn.html#loss-functions
loss_fn = nn.CrossEntropyLoss()

# Optimizador a usar https://pytorch.org/docs/stable/optim.html

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Scheduler a usar para ajuste del learning rating

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1)



# Entrenamiento: una iteracción de forward y backward, osease BackProp

def train(dataloader, model, loss_fn,optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        #Calculamos en base a la  función de perdida
        pred = model(X)
        loss = loss_fn(pred,y) 

        #BackPropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss,current = loss.item(), (batch+1)*len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")

# Test : estimación del rendimiento teórico con datos de test

def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss,correct = 0,0
    with torch.no_grad(): #Desactiva el Descenso por gradiente ya que NO es entrenamiento
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy
    

def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    accuracy = correct / size
    print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss, accuracy

#Backprop: con un número de épocas dado

test_accuracies = []
test_losses = []

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)

    val_loss, val_acc = validate(val_loader, model, loss_fn)
    
    scheduler.step(val_loss)  # reduce lr si val_loss no mejora
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')  # guardamos el mejor modelo
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping: sin mejora en validación.")
            break
    #Solo se hace al final el test con los datos completamente nuevos
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = test(test_loader, model, loss_fn)
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)

#Grabación del modelo final:
torch.save(model.state_dict(),"catdogs.pth")
print("Saved PyTorch Model State to catdogs.pth")


# Graficar accuracy y loss juntos
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Precisión (%)', color=color)
ax1.plot(range(1, epochs+1), test_accuracies, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # segundo eje y
color = 'tab:red'
ax2.set_ylabel('Pérdida (Loss)', color=color)
ax2.plot(range(1, epochs+1), test_losses, marker='x', linestyle='--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Precisión y Pérdida en Test por Época')
fig.tight_layout()
plt.grid(True)
plt.show()


