
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
def get_physics_loss(model, x, t, nu=0.01/np.pi):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = u_t + u * u_x - nu * u_xx
    return torch.mean(f**2)
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x_col = torch.linspace(-1, 1, 100).view(-1, 1)
t_col = torch.linspace(0, 1, 100).view(-1, 1)
X, T = torch.meshgrid(x_col.squeeze(), t_col.squeeze(), indexing='ij')
x_flat = X.reshape(-1, 1)
t_flat = T.reshape(-1, 1)
x_ic = torch.linspace(-1, 1, 100).view(-1, 1)
t_ic = torch.zeros_like(x_ic)
u_ic_target = -torch.sin(np.pi * x_ic)

print("Training started")
for epoch in range(2001):
    optimizer.zero_grad()
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_target)**2)
    loss_pde = get_physics_loss(model, x_flat, t_flat)
    total_loss = loss_ic + loss_pde
    total_loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
print("Training over, Plotting results")
with torch.no_grad():
    u_pred = model(x_flat, t_flat).reshape(100, 100).numpy()

plt.figure(figsize=(8, 4))
plt.imshow(u_pred, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='rainbow')
plt.colorbar(label='Velocity (u)')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('PINN Solution for Burgers\' Equation')
plt.show()
