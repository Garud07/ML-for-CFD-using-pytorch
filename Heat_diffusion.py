import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
def cal_physics_loss(model, x, t, alpha=0.1):
    x.requires_grad_(True)
    t.requires_grad_(True)
    T = model(x, t)
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_t = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    f = T_t - alpha * T_xx 
    return torch.mean(f**2)
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x_col = torch.linspace(-1, 1, 100).view(-1, 1)
t_col = torch.linspace(0, 1, 100).view(-1, 1)
X, Ti = torch.meshgrid(x_col.squeeze(), t_col.squeeze(), indexing='ij')
x_flat = X.reshape(-1, 1)
t_flat = Ti.reshape(-1, 1)
x_ic = torch.linspace(-1, 1, 100).view(-1, 1)
t_ic = torch.zeros_like(x_ic)            
T_ic_target = -torch.sin(np.pi * x_ic)
t_bc = torch.linspace(0, 1, 100).view(-1, 1)
x_bc_left = -torch.ones_like(t_bc)
x_bc_right = torch.ones_like(t_bc)
print("Training started...")
for epoch in range(2001):
    optimizer.zero_grad()
    T_ic_pred = model(x_ic, t_ic) 
    loss_ic = torch.mean((T_ic_pred - T_ic_target)**2)
    T_left_pred = model(x_bc_left, t_bc)
    T_right_pred = model(x_bc_right, t_bc)
    loss_bc = torch.mean(T_left_pred**2) + torch.mean(T_right_pred**2)
    loss_pde = cal_physics_loss(model, x_flat, t_flat) 
    total_loss = loss_ic + loss_pde + loss_bc
    total_loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}")
print("Training over, Plotting results...")
with torch.no_grad():
    T_pred = model(x_flat, t_flat).reshape(100, 100).numpy()
plt.figure(figsize=(8, 4))
plt.imshow(T_pred, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='rainbow') 
plt.colorbar(label='Temperature (K)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('PINN Solution for 1D Heat Equation (With BCs)')
plt.show()